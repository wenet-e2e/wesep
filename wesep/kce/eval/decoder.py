import os
import copy
import yaml
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wesep.dataset.kce_dataset import build_kce_dataset as build_dataset
from wesep.utils.kce_utils import read_list
from wesep.models.aed_kws_asr_phone import AEDKWSASRPhone
from wesep.kce.checkpoint_converter import convert_checkpoint

from wesep.utils.kce_utils import (
    setup_logger,
    batch_data_to_device,
    find_keyword_sublist_span,
    log_json_info,
)
from wesep.kce.eval.metrics import KWSEvaluator, ASREvaluator
from wesep.kce.eval.attention import AttentionAnalyzer
from wesep.kce.eval.threshold import ThresholdSweeper


class Decoder:
    def __init__(
        self,
        model_class: torch.nn.Module,
        checkpoint_path: str,
        inference_result_dir: str,
        test_data_list: str,
        test_data_config: Dict,
        rank: int = 0,
        world_size: int = 1,
        use_cuda: bool = False,
        attention_config: Optional[Dict] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.test_data_list = test_data_list
        self.enable_attention = attention_config is not None

        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.data_config = self._load_yaml(os.path.join(checkpoint_dir, "data.yaml"))
        self.model_config = self._load_yaml(os.path.join(checkpoint_dir, "model.yaml"))

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.model = self._load_model(model_class, checkpoint_path)

        self._prepare_data_config(test_data_config)

        self.inference_result_dir = inference_result_dir
        self.log_dir = os.path.join(inference_result_dir, "log")
        self.attention_map_save_dir = os.path.join(inference_result_dir, "attention_maps")
        self.analysis_save_dir = os.path.join(inference_result_dir, "analysis_results")
        self._create_directories()

        setup_logger(self.log_dir, rank)

        self.kws_evaluator = KWSEvaluator()
        self.asr_evaluator = ASREvaluator()
        self.attention_analyzer = AttentionAnalyzer(attention_config)
        self.threshold_sweeper = ThresholdSweeper(save_dir=self.analysis_save_dir)

        self.build_dataloader()

    # ------------------------------------------------------------------
    #  Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str) -> Dict:
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _load_model(self, model_class, checkpoint_path):
        model = model_class(**self.model_config).to(self.device)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)["model"]

        mapping_rules = [
            (r"\bau_trans\b", "speech_input_projection"),
            (r"\bau_transformer\b", "speech_transformer"),
            (r"\bkw_trans\b", "keyword_input_projection"),
            (r"\bbpe_asr_crit\b", "asr_bpe_criterion"),
            (r"\bphn_asr_crit\b", "asr_phn_criterion"),
            (r"\bkw_transformer\b", "keyword_transformer"),
        ]
        converted = convert_checkpoint(state_dict, model, mapping_rules=mapping_rules)
        model.load_state_dict(converted)
        model.eval()
        return model

    def _prepare_data_config(self, test_data_config: Dict):
        self.data_config["training_data_conf"] = copy.deepcopy(self.data_config)
        for key, sub_config in test_data_config.items():
            self.data_config[key] = sub_config

        self.data_config.pop("sv_config", None)
        if "sph_config" in self.data_config:
            self.data_config["speech_config"] = self.data_config.pop("sph_config")

        special = self.data_config["keyword_selection_conf"]["config"]["special_token"]
        self.punk = special["punk"]
        special["with_trans"] = False

    def _create_directories(self):
        for d in [self.log_dir, self.attention_map_save_dir, self.analysis_save_dir]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def build_dataloader(self):
        num_workers = self.data_config.get("num_workers", 0)
        tt_list = read_list(self.test_data_list, split_cv=False)[self.rank :: self.world_size]
        self.num_samples = len(tt_list)
        self.total_batches = int(np.ceil(self.num_samples / self.data_config["batch_size"]))

        self.tt_set = build_dataset(
            data_conf=self.data_config, data_list=tt_list, tag_of_dataset="test"
        )
        self.tt_loader = DataLoader(
            dataset=self.tt_set, batch_size=None, num_workers=num_workers
        )

    # ------------------------------------------------------------------
    #  Metadata helpers
    # ------------------------------------------------------------------

    def postprocess_meta_data(self, meta_dict: Dict) -> Dict:
        assert meta_dict["num_corrupt"][0] == 1, "Only support 1 interferer"

        if "corruption_material" in meta_dict:
            speech_interference_meta = []
            for bidx in range(len(meta_dict["corruption_material"])):
                interference = meta_dict["corruption_material"][bidx][1]
                for key in ["kw_candidate", "b_kw_candidate", "bpe_label", "phn_label"]:
                    interference.pop(key, None)
                interference["text"] = " ".join(interference["text"])
                speech_interference_meta.append(interference)
            meta_dict.pop("corruption_material")
        else:
            speech_interference_meta = []

        meta_dict["text"] = [" ".join(s) for s in meta_dict["text"]]

        keyword_positions, keywords = [], []
        for keyphrase, word_labels, match_case in zip(
            meta_dict["keyword"], meta_dict["phn_label_list"], meta_dict["match_case"]
        ):
            keyphrase = keyphrase.tolist()[2:-2]
            keywords.append(keyphrase)
            if match_case == "zero_match":
                keyword_positions.append((-1, -1))
            else:
                pos = find_keyword_sublist_span(keyphrase, word_labels)
                if pos is None:
                    raise ValueError("Keyword position not found")
                keyword_positions.append(pos)

        result = {
            **meta_dict,
            "keyword": keywords,
            "keyword_position": keyword_positions,
            "interference": speech_interference_meta,
        }
        result.pop("phn_label_list", None)
        return result

    def map_filler_label_for_neg(
        self, ground_truths, ctc_alignments, kws_targets
    ):
        batch_size = len(kws_targets)
        all_zeros, include_punk = 0, 0
        for b in range(batch_size):
            if kws_targets[b][0] == 1:
                continue
            ground_truths[b] = [self.punk]
            if self.punk in ctc_alignments[b] or sum(ctc_alignments[b]) == 0:
                all_zeros += int(sum(ctc_alignments[b]) == 0)
                include_punk += int(self.punk in ctc_alignments[b])
                ctc_alignments[b] = [self.punk]
        return ground_truths, ctc_alignments, all_zeros, include_punk

    # ------------------------------------------------------------------
    #  Main evaluation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, extract_embedding=False, embedding_save_dir=None):
        asr_flat = ASREvaluator.build_flat_cache()
        kws_cache = KWSEvaluator.build_flat_cache()
        att_flat = self._build_att_cache()

        neg_total = 0
        neg_correct = 0
        neg_all_zeros = 0
        neg_include_punk = 0

        if extract_embedding and embedding_save_dir:
            os.makedirs(embedding_save_dir, exist_ok=True)

        pbar = tqdm(enumerate(self.tt_loader), total=self.total_batches, desc="eval")
        for batch_id, data in pbar:
            torch.cuda.empty_cache()

            forward_data, meta_dict, _ = data
            forward_data = batch_data_to_device(forward_data, self.device)
            meta_dict = self.postprocess_meta_data(meta_dict)

            gts, ctc_alis, kws_targets, stats = self.model.evaluate(
                forward_data, return_embedding=extract_embedding)

            # Save speaker embeddings if requested
            if extract_embedding and embedding_save_dir and 'speaker_emb' in stats:
                speaker_embs = stats['speaker_emb']       # (B, 256)
                per_layer_embs = stats.get('per_layer_emb', None)  # 9 × (B, 256)
                for bidx, utt_key in enumerate(meta_dict.get('key', [str(batch_id)])):
                    save_dict = {
                        'key': utt_key,
                        'speaker_emb': speaker_embs[bidx].cpu(),
                    }
                    if per_layer_embs is not None:
                        save_dict['per_layer_emb'] = torch.stack(
                            [layer[bidx].cpu() for layer in per_layer_embs]
                        )  # (9, 256)
                    torch.save(
                        save_dict,
                        os.path.join(embedding_save_dir, f'{utt_key}.pt'),
                    )

            # KWS eval: run BEFORE map_filler_label_for_neg (needs original ctc_alis)
            self.kws_evaluator.evaluate_batch(meta_dict, ctc_alis, gts, kws_targets, kws_cache)

            if self.enable_attention:
                self.attention_analyzer.analyze_batch(
                    stats,
                    meta_dict["match_case"],
                    att_flat,
                    meta_dict,
                    enable=True,
                )

            gts, ctc_alis, zeros, punk = self.map_filler_label_for_neg(
                gts, ctc_alis, kws_targets
            )

            neg_idx = [i for i in range(len(kws_targets)) if kws_targets[i][0] == 0]
            neg_total += len(neg_idx)
            neg_correct += sum(1 for a in ctc_alis if a == [self.punk])
            neg_all_zeros += zeros
            neg_include_punk += punk

            self.asr_evaluator.evaluate_batch(meta_dict, ctc_alis, gts, asr_flat)

        self._print_results(asr_flat, kws_cache, neg_total, neg_correct, neg_all_zeros, neg_include_punk)

    @staticmethod
    def _build_att_cache():
        metrics = [
            "sharpness", "top_5p", "normalized_entropy",
            "spearman_corr", "high_att_ratio",
            "local_dtw", "local_cross_correlation",
        ]
        return {case: {m: [] for m in metrics} for case in KWSEvaluator.ALL_CASES}

    # ------------------------------------------------------------------
    #  Result output
    # ------------------------------------------------------------------

    def _print_results(self, asr_cache, kws_cache, neg_total, neg_correct, neg_zeros, neg_punk):
        self._print_kws(kws_cache)
        self._print_asr(asr_cache)
        self._print_neg_stats(neg_total, neg_correct, neg_zeros, neg_punk)

    def _print_kws(self, kws_cache):
        logging.info("=" * 50)
        logging.info("KWS Evaluation (Keyword Spotting)")
        for case in ["one_match", "multiple_match", "zero_match"]:
            stats = kws_cache.get(case, {})
            tp = stats.get("tp", 0)
            fp = stats.get("fp", 0)
            tn = stats.get("tn", 0)
            fn = stats.get("fn", 0)
            total_pos = stats.get("total_positive", 0)
            total_neg = stats.get("total_negative", 0)
            if tp + fn == 0 and tn + fp == 0:
                continue
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            logging.info("  %-18s  TP=%-5d FP=%-5d TN=%-5d FN=%-5d  Recall=%.1f%%  Prec=%.1f%%  F1=%.1f%%",
                         case, tp, fp, tn, fn, recall, precision, f1)

    def _print_asr(self, asr_cache):
        logging.info("=" * 50)
        logging.info("ASR Evaluation (Phoneme Error Rate)")
        for case in ["one_match", "zero_match", "multiple_match"]:
            per_list = asr_cache[case].get("per", [])
            if not per_list:
                continue
            avg = np.mean(per_list)
            logging.info("  %-18s  PER=%.1f%%  (n=%d)", case, avg * 100, len(per_list))

    @staticmethod
    def _print_neg_stats(total, correct, zeros, punk):
        logging.info("=" * 50)
        logging.info("Negative-sample filler mapping")
        rate = correct / total if total > 0 else 0.0
        logging.info("  correct=%d/%d (%.1f%%)  all-zeros=%d  include-punk=%d",
                     correct, total, rate * 100, zeros, punk)

    # ------------------------------------------------------------------
    #  Entry point
    # ------------------------------------------------------------------

    def run(self, extract_embedding=False, embedding_save_dir=None):
        self.eval(extract_embedding=extract_embedding, embedding_save_dir=embedding_save_dir)
