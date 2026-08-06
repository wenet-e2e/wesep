"""
Extract KCE speaker embeddings from LibriMix mixtures for TSE inference.

Supports flexible embedding dimensions via per-layer concatenation:
  - dim=256:  SV-pooled embedding (1 layer, learnable_weights pooling)
  - dim=512:  2 per-layer embeddings concatenated
  - dim=768:  3 per-layer embeddings concatenated
  - dim=1024: 4 per-layer embeddings concatenated (e.g. layers [2,4,6,8])

Usage:
  python src/kce/bin/extract_embed_for_tse.py \
    --checkpoint_path exp/.../kwatt_asr_149.pt \
    --input_jsonl /public/TSEASR/LibriMix/LibriMixData/Libri2Mix/wav16k/max/libri2mix_test_both.jsonl \
    --output_dir exp/.../tse_embeddings \
    --embedding_dim 1024 \
    --use_gpu_id 0
"""

import argparse
import json
import os
import sys
import logging
try:
    import g2p_en
except ImportError:
    g2p_en = None

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from tqdm import tqdm

# Add KCE to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KCE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if KCE_ROOT not in sys.path:
    sys.path.insert(0, KCE_ROOT)

from wesep.kce.model.AEDKWSASRPhone import AEDKWSASRPhone
import wesep.kce.model.NetModules as NM


# ------------------------------------------------------------------
#  Config
# ------------------------------------------------------------------

DEFAULT_LAYER_INDICES = {
    256:  None,           # Use SV-pooled embedding (learnable_weights)
    512:  [4, 8],         # 2 layers
    768:  [2, 5, 8],      # 3 layers
    1024: [2, 4, 6, 8],   # 4 layers
}

# Special tokens (must match training config)
SPECIAL_TOKENS = {
    'sok': 5002,
    'eok': 5003,
    'unk': 1,
    'psok': 71,
    'peok': 72,
    'punk': 73,
    'sos': 2,
    'eos': 5004,
}

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'


# ------------------------------------------------------------------
#  Phoneme conversion
# ------------------------------------------------------------------

def load_phoneme_resources(lexicon_path=None, p2idx_path=None):
    """Load word2lexicon and phoneme2int mappings.

    IMPORTANT: The p2idx from phoneme2int.txt (NOT g2p_en's built-in) is the
    canonical mapping used during KCE training. The KCE model was trained with:
      - phoneme tokens: 0-70 (from phoneme2int.txt)
      - special tokens: psok=71, peok=72, punk=73
      - total: num_phn_token=74

    g2p_en is used ONLY for grapheme-to-phoneme string conversion, then
    phoneme strings are remapped to indices using phoneme2int.txt's p2idx.
    """
    word2lexicon = {}
    p2idx = {}

    # Load lexicon (pre-mapped using phoneme2int.txt indices)
    if lexicon_path and os.path.exists(lexicon_path):
        with open(lexicon_path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    word, ints = parts
                    word2lexicon[word] = [int(x) for x in ints.split()]

    # Load phoneme2int.txt mapping (CANONICAL for KCE)
    if p2idx_path and os.path.exists(p2idx_path):
        with open(p2idx_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    p2idx[parts[0]] = int(parts[1])

    if not p2idx:
        logging.warning("No p2idx loaded! Token indices may be wrong.")

    return word2lexicon, p2idx


def text_to_phonemes(text, word2lexicon, p2idx):
    """Convert text string to flat phoneme index list using KCE's token space.

    Uses word2lexicon (pre-mapped) for known words; falls back to g2p_en for
    phoneme string conversion, then remaps via p2idx (phoneme2int.txt).
    """
    # Normalize
    text = text.strip()
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    text = text.lower()
    words = text.split()

    phonemes = []
    # Create g2p object once for fallback (phoneme string conversion only)
    g2p_obj = g2p_en.G2p() if g2p_en is not None else None

    for word in words:
        if word in word2lexicon:
            phonemes.extend(word2lexicon[word])
        elif g2p_obj is not None:
            # g2p_en for phoneme strings, then remap via phoneme2int.txt p2idx
            phones = g2p_obj(word)
            for p in phones:
                if p in p2idx:
                    phonemes.append(p2idx[p])
                else:
                    # Unknown phoneme -> use punk (73)
                    phonemes.append(SPECIAL_TOKENS['punk'])
        else:
            # Cannot convert, skip word
            continue

    return phonemes


def make_keyword_tokens(phonemes, kw_start=0, kw_len=6):
    """Wrap phoneme segment with KCE special tokens.

    KCE uses psok (71) and peok (72) for phoneme-level keywords.
    The sok/eok (5002/5003) are BPE-level tokens, NOT used for phoneme keywords.

    Format: [psok, phn_1, phn_2, ..., phn_n, peok]
    All indices stay within [0, 73] (num_phn_token=74).
    """
    kw_phonemes = phonemes[kw_start:kw_start + kw_len]
    if not kw_phonemes:
        return None

    kw_label = [SPECIAL_TOKENS['psok']] + kw_phonemes + [SPECIAL_TOKENS['peok']]

    return torch.tensor(kw_label, dtype=torch.long)


# ------------------------------------------------------------------
#  Model loading
# ------------------------------------------------------------------

def load_kce_model(checkpoint_path, device):
    """Load the KCE model from checkpoint."""
    exp_dir = os.path.dirname(checkpoint_path)

    with open(os.path.join(exp_dir, 'model.yaml')) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    model = AEDKWSASRPhone(**model_config).to(device)
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    if 'model' in state_dict:
        state_dict = state_dict['model']

    # Handle key renaming (same as Decoder._load_model)
    import re
    mapping_rules = [
        (r"\bau_trans\b", "speech_input_projection"),
        (r"\bau_transformer\b", "speech_transformer"),
        (r"\bkw_trans\b", "keyword_input_projection"),
        (r"\bbpe_asr_crit\b", "asr_bpe_criterion"),
        (r"\bphn_asr_crit\b", "asr_phn_criterion"),
        (r"\bkw_transformer\b", "keyword_transformer"),
    ]
    from wesep.kce.model.checkpoint_converter import convert_checkpoint
    converted = convert_checkpoint(state_dict, model, mapping_rules=mapping_rules)
    model.load_state_dict(converted)
    model.eval()
    return model


# ------------------------------------------------------------------
#  Embedding extraction
# ------------------------------------------------------------------

@torch.no_grad()
def extract_embedding(model, audio_path, keyword_tokens, device,
                      frontend_conf, embedding_dim=256, layer_indices=None):
    """
    Extract speaker embedding from an audio file.

    Args:
        model: KCE model (AEDKWSASRPhone)
        audio_path: path to wav file
        keyword_tokens: (kw_len,) tensor of keyword token IDs
        device: torch device
        frontend_conf: fbank config dict
        embedding_dim: desired output dimension (256/512/768/1024)
        layer_indices: explicit layer indices to concat (overrides DEFAULT_LAYER_INDICES)

    Returns:
        embedding: (embedding_dim,) tensor
        per_layer_emb: 9 × (256,) - all per-layer pooled embeddings
        speaker_emb: (256,) - SV-pooled embedding
    """
    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = wav.squeeze(0)  # (T,)

    # Compute fbank
    fbank = kaldi.fbank(wav.unsqueeze(0), **frontend_conf)  # (T, 80)
    fbank_len = torch.tensor([fbank.shape[0]], device=device)
    fbank = fbank.unsqueeze(0).to(device)  # (1, T, 80)

    # Keyword
    kw_label = keyword_tokens.unsqueeze(0).to(device)  # (1, KwLen)
    kw_len = torch.tensor([kw_label.shape[1]], device=device)

    # Compute lengths after conv
    sph_len = NM.BaseConv.compute_dim_reduction(fbank_len, 3, 2, 0, 1)
    sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)

    b = fbank.shape[0]
    sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
    kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
    cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

    # Speech conv + projection
    speech_feature = model.au_conv(fbank.unsqueeze(1))
    b, c, t, d = speech_feature.size()
    speech_feature = model.au_conv_trans(
        speech_feature.transpose(1, 2).contiguous().view(b, t, c * d))
    speech_feature = model.speech_input_projection(speech_feature)

    # Keyword embedding
    keyword_feature = model.phn_emb(kw_label.to(torch.long))
    keyword_feature = model.keyword_input_projection(keyword_feature)

    # Position encoding
    speech_feature = model.speech_pe_module(speech_feature)
    keyword_feature = model.keyword_pe_module(keyword_feature)

    # Forward transformers
    keyword_feature = model.forward_transformer(
        model.keyword_transformer, keyword_feature, attention_mask=kw_mask)

    speech_feature, (_, sph_emb_list) = model.forward_transformer(
        model.speech_transformer, speech_feature,
        attention_mask=sph_mask,
        cross_hidden_states=(keyword_feature, keyword_feature, cross_mask),
        analyse=True)

    # sph_emb_list[0] = list of 9 tensors, each (1, T', 256)
    raw_emb_list = sph_emb_list[0]

    # Per-layer time-pooled embeddings
    per_layer_emb = []
    for layer_hidden in raw_emb_list:
        pooled = layer_hidden[0, :sph_len[0], :].mean(0)  # (256,)
        per_layer_emb.append(pooled.cpu())

    # SV-pooled speaker embedding
    if model.use_sv:
        speaker_emb = model.sv_ce_crit.forward_pooling(raw_emb_list, sph_len)
        speaker_emb = speaker_emb[0].cpu()  # (256,)
    else:
        speaker_emb = per_layer_emb[-1]

    # Build output embedding based on desired dimension
    if embedding_dim == 256:
        output_emb = speaker_emb  # Use SV-pooled
    else:
        if layer_indices is not None:
            selected = [per_layer_emb[i] for i in layer_indices]
        else:
            default_indices = DEFAULT_LAYER_INDICES.get(embedding_dim,
                                                        DEFAULT_LAYER_INDICES[1024])
            selected = [per_layer_emb[i] for i in default_indices]
        output_emb = torch.cat(selected, dim=0)  # (embedding_dim,)

    return output_emb.cpu(), per_layer_emb, speaker_emb.cpu()


# ------------------------------------------------------------------
#  Main pipeline
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract KCE embeddings for TSE inference')
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to KCE model checkpoint (.pt)')
    parser.add_argument('--input_jsonl', required=True,
                        help='Path to LibriMix test JSONL')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save extracted embeddings')
    parser.add_argument('--embedding_dim', type=int, default=1024,
                        choices=[256, 512, 768, 1024],
                        help='Output embedding dimension (default: 1024)')
    parser.add_argument('--use_gpu_id', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--keyword_length', type=int, default=6,
                        help='Length of keyword in phonemes')
    parser.add_argument('--lexicon_path', type=str,
                        default='examples/librimix/dae-tse/data/text_cue/word2lexicon.txt',
                        help='Path to word2lexicon.txt')
    parser.add_argument('--p2idx_path', type=str,
                        default='examples/librimix/dae-tse/data/text_cue/phoneme2int.txt',
                        help='Path to phoneme2int.txt')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of samples (0 = all)')
    parser.add_argument('--layer_indices', type=int, nargs='+', default=None,
                        help='Explicit layer indices to concatenate, e.g. --layer_indices 5 6 7 8. '
                             'Overrides the default selection for embedding_dim.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples that already have embeddings saved')

    args = parser.parse_args()

    # Setup
    device = torch.device(f'cuda:{args.use_gpu_id}') if args.use_gpu_id >= 0 \
        else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Load KCE model
    logger.info(f"Loading KCE model from {args.checkpoint_path}")
    model = load_kce_model(args.checkpoint_path, device)
    model.eval()

    # Load frontend config
    exp_dir = os.path.dirname(args.checkpoint_path)
    with open(os.path.join(exp_dir, 'data.yaml')) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    frontend_conf = data_config['speech_config']['feats_config']

    # Load phoneme resources
    logger.info("Loading phoneme resources...")
    word2lexicon, p2idx = load_phoneme_resources(
        args.lexicon_path, args.p2idx_path)
    logger.info(f"  word2lexicon: {len(word2lexicon)} entries")
    logger.info(f"  p2idx: {len(p2idx)} entries")

    # Load input JSONL
    with open(args.input_jsonl) as f:
        input_data = [json.loads(line) for line in f]

    if args.limit > 0:
        input_data = input_data[:args.limit]

    actual_layers = args.layer_indices if args.layer_indices is not None \
                    else DEFAULT_LAYER_INDICES.get(args.embedding_dim, 'SV-pooled')
    logger.info(f"Processing {len(input_data)} samples, "
                f"embedding_dim={args.embedding_dim} "
                f"(layers: {actual_layers})")

    # Extract embeddings for each target speaker in each mixture
    results = []
    skipped = 0
    for item in tqdm(input_data, desc="Extracting embeddings"):
        audio_path = item['audio']['path']
        # LibriMix JSONL paths have "./dataset/" prefix; map to actual location
        if audio_path.startswith('./dataset/'):
            audio_path = audio_path.replace('./dataset/', '/public/TSEASR/', 1)
        elif audio_path.startswith('./'):
            jsonl_dir = os.path.dirname(os.path.abspath(args.input_jsonl))
            audio_path = os.path.normpath(os.path.join(jsonl_dir, audio_path))

        sentences = item.get('sentences', [])
        if len(sentences) < 2:
            logger.warning(f"Fewer than 2 sentences for {audio_path}, skipping")
            continue

        # Extract utterance key from audio path
        utt_key = os.path.splitext(os.path.basename(audio_path))[0]

        # Process each target speaker
        for spk_idx, sentence in enumerate(sentences):
            target_text = sentence['text']
            out_file = os.path.join(args.output_dir, f'{utt_key}_spk{spk_idx}.pt')

            if args.skip_existing and os.path.exists(out_file):
                skipped += 1
                continue

            # Convert text to phonemes and make keyword
            phonemes = text_to_phonemes(target_text, word2lexicon, p2idx)
            if not phonemes:
                logger.warning(f"No phonemes for {utt_key}_spk{spk_idx}: '{target_text[:50]}...'")
                continue
            kw_tokens = make_keyword_tokens(phonemes, kw_start=0,
                                            kw_len=args.keyword_length)
            if kw_tokens is None:
                continue

            # Extract embedding
            try:
                output_emb, per_layer_emb, speaker_emb = extract_embedding(
                    model, audio_path, kw_tokens, device,
                    frontend_conf, args.embedding_dim,
                    layer_indices=args.layer_indices)
            except Exception as e:
                logger.error(f"Error processing {utt_key}_spk{spk_idx}: {e}")
                continue

            # Save
            save_dict = {
                'utt_key': f'{utt_key}_spk{spk_idx}',
                'audio_path': audio_path,
                'speaker_id': item.get('speakers', ['unknown', 'unknown'])[spk_idx]
                              if spk_idx < len(item.get('speakers', [])) else 'unknown',
                'target_text': target_text,
                'embedding': output_emb,
                'embedding_dim': args.embedding_dim,
                'speaker_emb_256': speaker_emb,
                'per_layer_embs': torch.stack(per_layer_emb),  # (9, 256)
            }
            torch.save(save_dict, out_file)
            results.append({
                'utt_key': f'{utt_key}_spk{spk_idx}',
                'out_file': out_file,
                'emb_dim': args.embedding_dim,
            })

    # Save metadata
    meta_file = os.path.join(args.output_dir, 'extraction_meta.json')
    with open(meta_file, 'w') as f:
        json.dump({
            'checkpoint_path': args.checkpoint_path,
            'input_jsonl': args.input_jsonl,
            'embedding_dim': args.embedding_dim,
            'layer_indices': args.layer_indices if args.layer_indices is not None
                            else DEFAULT_LAYER_INDICES.get(args.embedding_dim),
            'num_samples': len(results),
            'skipped': skipped,
            'results': results,
        }, f, indent=2)

    logger.info(f"Extracted {len(results)} embeddings (skipped {skipped})")
    logger.info(f"Saved to {args.output_dir}")
    logger.info(f"Metadata: {meta_file}")


if __name__ == '__main__':
    main()
