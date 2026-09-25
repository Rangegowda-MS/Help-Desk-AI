# Model card: HelpDesk AI demo

## Intended use
Educational implementation and demonstration of LSTM sequence-to-sequence modeling. It is not a live customer-support service. There are no external model APIs, account connections, payment operations, or retrieval shortcuts.

## Data provenance
240 original synthetic English question/answer pairs, written for this project, across 30 fictional support topics. No private customer records or Cornell movie text are bundled. The assignment explicitly permits a custom support FAQ dataset. The CSV can be edited and replaced.

## Architecture
Trainable 64-dimensional word embeddings; one-layer 128-dimensional LSTM encoder and decoder; masked Bahdanau attention; dropout 0.15. Up to 15 cleaned words plus EOS. Vocabulary built on training text only. Adam learning rate 0.003, batch size 32, seed 42, clip norm 1.0.

## Actual bundled training run
Requested 100 epochs with early-stopping patience 15. Completed 36 epochs; selected epoch 21. Split: 192 train / 24 validation / 24 test. Vocabulary: 268 tokens. Training ran on CPU.

| Metric | Actual value |
|---|---:|
| Best validation cross-entropy | 0.5441 |
| Test cross-entropy | 0.5259 |
| Test perplexity | 1.6920 |
| Greedy exact reference match | 54.17% (13/24) |
| Test generations reaching EOS | 100% |

Perplexity uses full teacher forcing; exact match uses free-running greedy decoding. Review every generated answer in `artifacts/predictions.csv`. Lower loss does not ensure correct support advice.

## Evaluation limitations
Only 24 test questions. Normalized duplicate questions do not cross splits, but related paraphrases and identical target answers do. Metrics are optimistic relative to unseen domains and real-world queries. No no-attention ablation was run, so this artifact does not experimentally prove an attention improvement. Beam search is implemented but no quality improvement is claimed.

## Interaction limitations
One question at a time, English only, short inputs, limited vocabulary, no genuine multi-turn memory. The UI stores a transcript for the session but does not condition generation on it. Unknown words, unrelated questions, and long questions can produce inaccurate or generic replies. Synthetic support instructions must not be treated as real company policy. Attention is not calibrated confidence or a proof of reasoning.

## Verification
Unit checks cover cleaning, unknown words, prompt split separation, attention padding masks, normalized weights, forward shapes, finite gradients, and padding-invariant encoder state. Streamlit AppTest exercised launch, greeting, beam-search password-reset reply, and conversation reset. All notebook code cells were executed sequentially in-process with real outputs captured because the build environment could not start a socket-based Jupyter kernel; a normal interactive kernel should be used when reviewing locally.

## Reproduction
Run `python train.py --epochs 100 --patience 15`, then `python evaluate.py`. Exact floating-point results may vary by platform and PyTorch release. `artifacts/config.json` includes the input CSV SHA-256. `docs/TESTED_VERSIONS.txt` records the build environment's main package versions.
