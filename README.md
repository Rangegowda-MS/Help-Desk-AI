# HelpDesk AI — Seq2Seq Chatbot with Bahdanau Attention

A runnable educational implementation of the supplied Persevex **Chatbot using Sequence-to-Sequence (Seq2Seq) Model** brief, with a small set of advanced extensions.

## Start on Windows (PowerShell)

Install Python 3.11 or 3.12. Extract this ZIP and open a terminal inside the `HelpDesk-AI` folder (the folder containing `app.py`). Run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in the terminal, normally http://localhost:8501. Virtual-environment activation is not required. The included trained checkpoint lets you start immediately. No API key is needed.

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Updated interface

The interface now includes a navy-and-mint layout, clickable example questions, chat export, styled attention maps, an interactive loss chart, and a filter for held-out answers that did not exactly match. `app.py` is split into named functions; all custom CSS lives in `assets/style.css`. See `docs/MAKE_IT_YOURS.md` for a code map and a personal experiment log.

For an existing installation, stop Streamlit with Ctrl+C, replace the old project files with this ZIP's project files, keep your `.venv` folder, and run the app again. Copy the `assets` folder and `.streamlit/config.toml` as well as `app.py`. If you have edited your dataset or trained new weights, back up those files before replacing anything.

## What matches the brief?

| PDF requirement | Implementation / evidence |
|---|---|
| Cornell **or** custom support FAQ dataset | `data/support_faq.csv`: 240 original synthetic support pairs, 30 topics |
| Text cleaning; maximum 10–15 words | `helpdesk/data.py`: lowercase, remove non-alphanumerics, trim to 15 words |
| Vocabulary and SOS/EOS/PAD/UNK | Training-only vocabulary; padded batches; `artifacts/vocabulary.json` |
| Encoder LSTM | `helpdesk/model.py`: embeddings and packed LSTM |
| Decoder LSTM | `helpdesk/model.py`: recurrent one-token decoding |
| Bahdanau or Luong attention | Masked Bahdanau additive attention, explicit implementation |
| Teacher forcing | `train.py`: ratio scheduled from 1.0 toward 0.5 |
| Categorical cross-entropy | PyTorch CrossEntropyLoss with integer class targets; padding ignored |
| Autoregressive inference and EOS stop | `helpdesk/inference.py`; greedy is the default |
| Model notebook, preprocessing, architecture, loss plots | `notebooks/HelpDesk_AI.ipynb` with executed outputs |
| Saved encoder and decoder weights | `artifacts/encoder.pt`, `artifacts/decoder.pt`; combined `best.pt` |
| Streamlit or Flask messaging interface | `app.py`, Streamlit chat messages and input |

## A little more advanced

- Attention heatmaps for every generated response.
- Optional length-normalized beam search (width 3 or 5).
- Validation-based early stopping and best checkpoint selection.
- Dropout, gradient clipping, reproducible seeds, and padding-aware attention/loss.
- Training dashboard, test perplexity, held-out generated responses, exact-match and EOS rates.
- Conversation export and reset; input-length and unknown-word indicators.
- Automated checks of masking, gradient flow, vocabulary handling, and disjoint prompts.

## Train and evaluate yourself

The following commands use `python` as shorthand for the virtual-environment executable above.

```bash
python train.py --epochs 50 --output my_run
python evaluate.py --directory my_run
python -m unittest discover -s tests -v
```

To replace the checkpoint used by the app, train to the default `artifacts` directory:

```bash
python train.py --epochs 100 --patience 15
python evaluate.py
```

This overwrites the bundled run. Preserve a copy first if you need its original results. The bundled run uses the second command's training configuration. Outputs include CSV splits, loss history, a loss image, configuration, vocabulary, model weights, and metrics. The combined checkpoint stores architecture and vocabulary so inference uses the same mapping as training. `encoder.pt` and `decoder.pt` are exported from the same best validation checkpoint, not from a different epoch.

To load the separate weights, construct `Seq2Seq` with the architecture in `artifacts/config.json`, use the vocabulary length from `vocabulary.json`, then call each component's `load_state_dict(torch.load(path, weights_only=True))`.

For your own dataset, supply a CSV with `question` and `answer` columns:

```bash
python train.py --data data/my_support.csv --epochs 50 --output my_run
```

Use at least 20 distinct nonempty questions (hundreds or thousands of diverse, reviewed pairs are preferable). The supplied custom FAQ option fully follows the dataset choice in the PDF; Cornell is not required in addition. This implementation is English-only. Apostrophes and symbols are replaced with spaces. The 15-word cap is applied before EOS is appended.

## Notebook

Open `notebooks/HelpDesk_AI.ipynb` in VS Code with the Jupyter extension, select this virtual environment, and run all cells. It contains explanatory text, preprocessing, actual encoder/attention/decoder code, training logic, actual saved loss curves, test results, and an attention plot. By default it reads the included run. Set `RUN_TRAINING = True` in the training cell to retrain from the notebook.

## Files

- `helpdesk/data.py`: preprocessing, vocabulary, batching.
- `helpdesk/model.py`: neural network components.
- `helpdesk/inference.py`: checkpoint loading and decoding.
- `train.py`: training, validation, early stopping, exports.
- `evaluate.py`: held-out generation evaluation.
- `app.py`: Streamlit demo, organized into small UI functions.
- `assets/style.css`: responsive interface styling.
- `docs/MAKE_IT_YOURS.md`: personalization steps and code explanations.
- `notebooks/HelpDesk_AI.ipynb`: walkthrough with executed results.
- `artifacts/`: real checkpoint weights, splits, metrics, predictions, loss plot.
- `docs/SUBMISSION_GUIDE.md`: four-week execution plan and demo checklist.
- `docs/EXPLANATION_AND_VIVA.md`: beginner explanations and viva practice.
- `docs/MODEL_CARD.md`: scope, data provenance, limitations, and run results.

## Limitations you should explain in your presentation

This is a compact educational neural model, not a general-purpose assistant. Its dataset was authored synthetically for this project; the answers describe a fictional support service. The random split holds out distinct questions, but similar paraphrases and target answers occur across splits. Results therefore measure a narrow in-domain exercise, not generalization to new support topics. Incorrect, generic, or repetitive replies remain possible, particularly outside the training topics.

Chat history is a UI feature; the neural model conditions only on the current question. Multi-turn memory, real account actions, real-company knowledge, and production deployment are not included. Attention is a visualization of model weights, not a correctness guarantee. Perplexity is computed with full teacher forcing; free-running exact match is reported separately. Token probabilities are not calibrated confidence.

## Troubleshooting

- `No module named ...`: install requirements using the same Python executable that launches the app.
- `best.pt` missing: run `python train.py` from the project folder.
- Weak replies: review `artifacts/predictions.csv`, improve dataset diversity and label quality, retrain, and compare validation and held-out generations. More epochs alone may overfit.
- Port occupied: append `--server.port 8502` to the Streamlit command.
- Slow training: use a smaller hidden dimension (`--hidden-dim 64`) or a CUDA-compatible PyTorch installation with `--device cuda`. CPU is the default.

## References

- Assignment: supplied Persevex project export, pages 2–4.
- [PyTorch Seq2Seq and attention tutorial](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — conceptual reference; this project uses an LSTM implementation.
- [Streamlit chat input documentation](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input) — interface API reference.

Read the code, rerun the training, and explain your own observations when presenting the project.
