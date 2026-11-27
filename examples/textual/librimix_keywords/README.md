# LibriMix Textual-Cue TSE

This recipe trains BSRNN with the frozen DAE KCE. It reads existing Libri2Mix
mixtures and supports online or offline text-to-phoneme conversion.

## Input paths

Use `confs/tse_bsrnn_dae.yaml` for both paths. Set
`dataset_args.cue_processing.textual.input` to:

| Value | Phonemization |
| --- | --- |
| `text` (default) | CPU data-loader workers |
| `dae_phoneme` | Data preparation; workers read precomputed IDs |

`run.sh` reads the input choice from the configuration. Stage 1 generates
`json/word_text` or `json/dae_keyword_phoneme` cue indexes, respectively.
The reader rejects configuration/index mismatches. Both paths share
`dataset_args.cue_processing.textual.encoding`; the KCE always receives IDs.

```bash
# Prepare full transcripts plus the four official test conditions.
bash run.sh --stage 0 --stop_stage 1 \
  --librimix_root /path/to/Libri2Mix \
  --librispeech_root /path/to/LibriSpeech \
  --config confs/tse_bsrnn_dae.yaml --data data

bash run.sh --stage 3 --stop_stage 3 \
  --config confs/tse_bsrnn_dae.yaml --data data --gpus 0,1

# Select another official condition without regenerating data.
bash run.sh --stage 5 --stop_stage 5 \
  --config confs/tse_bsrnn_dae.yaml --data data --test_cue 2 \
  --checkpoint /path/to/checkpoint.pt
```

For offline input, set `input: dae_phoneme` in the same configuration and use
`--data data_phoneme` in preparation and subsequent stages. Keep the input choice
consistent with those cue indexes. Use `--exp_dir` to separate experiments when
comparing both paths. Changing the configuration alone does not convert existing
cue indexes; rerun Stage 1 for the selected path.

Stage 0 downloads KCE weights/sidecars, phoneme resources and the official four
test lists. Stage 1 uses `wav16k/min/{train-100,dev,test}` and LibriSpeech
`*.trans.txt` files. Stages 2-6 retain sharding, training, explicit checkpoint
averaging, inference and scoring; Stage 5 selects the requested test cue YAML.
No stage runs by default. `--test_cue` accepts `1`, `2`, `3`, `4`, or `full`,
with default `4`. It only selects testing cues; it does not change validation.

## Selection and reproducibility

Selection is configured separately for `train`, `val`, and `test`:

```yaml
selection:
  train:
    type: random_span
    word_range: [2, 6]
  val:
    type: random_span
    word_range: [2, 6]
    seed: 42
  # Omitted stages default to all available words.
```

`prefix` selects leading words, `random_span` selects a consecutive word span,
and `all` keeps every available word. The first two accept an inclusive
`word_range`; counts are capped by available words. With `seed`, both count and
position are determined from SHA256(seed, target key), independent of worker
count and input order. Without it, selection uses the ordinary worker RNG.
The run-level training seed still controls reproducibility across training runs.
Seed zero is valid. The former shared `eval` setting is no longer used.

The defaults use complete transcripts for training and validation, random spans
of 2-6 words in both stages, and fixed per-target validation sampling. Test has
no selection entry and consumes every word in its selected cue record.
A `selection.test` entry can be added for experiments with `--test_cue full`.
Using prefix on an official four-word cue does not reproduce the official
one-/two-/three-word conditions: the four official lists use different choices.

## Outputs and official cues

DAE weights and tokenizer resources always live in the recipe's `dae_kce/`
directory, independently of training outputs in `exp/` and the `--data` path:

```text
dae_kce/
  epoch_149.pt
  model.yaml
  data.yaml
  phoneme2int.txt
  word2lexicon.txt
  testset/kw-{1,2,3,4}_seed-42.jsonl
```


Official source lists remain in `dae_kce/testset/kw-{1,2,3,4}_seed-42.jsonl`.
Stage 1 converts their selected keywords into the WeSep cue indexes below.
Training and inference read only the converted indexes; they do not look up
keywords in the original DAE files.

```text
data/clean/
  textual_cue.json
  train-100/                  # dev has the same full-transcript structure
    samples.jsonl
    raw.list
    cues.yaml
    cues/textual.json
    cues/textual_phoneme.json  # offline input only
  test/
    samples.jsonl
    raw.list
    cues_full.yaml
    cues_kw1.yaml
    cues_kw2.yaml
    cues_kw3.yaml
    cues_kw4.yaml
    cues/textual.json
    cues/textual_phoneme.json  # offline input only
    cues/textual_kw1.json
    cues/textual_kw2.json
    cues/textual_kw3.json
    cues/textual_kw4.json
```

Records are keyed by `mixture_key::speaker_id` and contain `text`, `words`, and
`word_times` (null for LibriSpeech). Offline records add one `phn_label` group
per word. Generated cue JSON objects put each target record on its own line,
including all nested words/phonemes on that line. They remain ordinary JSON
objects readable by `json.load()`, not JSONL. Cue YAML resource paths are absolute. Samples and audio are shared
across test conditions. Validation does not generate four extra keyword lists.

Official source: https://github.com/GnafiY/DAE-TSE/tree/main/recipes/librimix/dae-tse/data/text_cue/testset

The converter stores only the officially selected words. Offline official
records preserve the selected published phoneme groups exactly; they are not
recomputed. Online records store the same keyword text and use the shared
tokenizer at runtime. Published phonemes can differ from re-encoding the same
text, so exact reproduction of the released phoneme inputs requires the
offline path. Full-transcript offline records use the shared tokenizer normally.
Existing previously generated `test/cues.yaml` files are not used by run.sh.

## Timing, tokenization and padding

When word times exist, the reader first filters against the absolute audio
window, then applies selection. `boundary: contained` keeps whole words;
`overlap` includes boundary-crossing words. Empty windows raise an error.
Untimed transcripts cannot identify which words survive min/chunk/batch cropping.

The CPU tokenizer in `wesep.modules.textual.tokenizer` is shared by workers,
offline preprocessing and direct text inference. It preserves per-original-word
groups after normalization. G2P is lazy for lexicon misses; install `g2p_en` and
prepare its language resources before starting workers.

Word groups contain ordinary phoneme IDs 1-70, without BOS/EOS. Collate uses
max-length constant padding with ID 0. KCE removes padding, adds BOS=71/EOS=72,
and uses per-target lengths to mask keyword attention. Its padding ID remains 0.

The default remains whole-utterance training with batch size 1. Multi-sample
text padding is supported, but audio still follows its own alignment: min can
crop longer waveforms, while max needs suitable true-length handling and loss
masking. This update does not change the audio-side contract. Head chunks can
be configured with `whole_utt: false` and `chunk_random_start: false`; without
word timings, keyword containment is not guaranteed.
