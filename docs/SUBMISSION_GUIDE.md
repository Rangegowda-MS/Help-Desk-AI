# Submission and demonstration guide

## Required deliverables

1. **Model notebook:** `notebooks/HelpDesk_AI.ipynb`. It covers cleaning, indexing, padding, model components, teacher forcing, training, and actual loss plots.
2. **Saved model:** include `artifacts/encoder.pt`, `decoder.pt`, `vocabulary.json`, `config.json`, and `best.pt` (the app uses the combined file).
3. **Messaging demo:** `app.py` and the `helpdesk` package. Include `requirements.txt` and README setup commands.

Submit the complete ZIP if allowed by your course portal. If it requires separate uploads, use the files above. No grading rubric or upload portal rules were supplied beyond the PDF, so check the portal's file limits yourself.

## Four-week execution plan from the brief

| Week | Required work | Your evidence |
|---|---|---|
| 1 | Data pipeline, question/answer pairs, vocabulary, padding | CSV, preprocessing notebook cells, example tensors |
| 2 | Encoder, attention layer, decoder | Explicit architecture code and attention explanation |
| 3 | Teacher forcing, training, convergence | Retraining log, history.csv, loss.png, checkpoint |
| 4 | Inference loop, app, testing, documentation | Streamlit demo, predictions.csv, tests, README |

The PDF gives a four-week plan but does not specify a calendar submission date.

## Suggested five-minute demo

- Explain the problem: generating short responses to common customer-support questions.
- Show the custom FAQ dataset and disclose that it is synthetic.
- Walk through cleaning, special tokens, and padding in the notebook.
- Explain encoder hidden/cell states, attention context, and one-token decoder outputs.
- Show the actual loss plot and discuss train/validation differences.
- Open the app and try a greeting, password reset, invoice question, and team invitation.
- Expand “Why these words? Explore attention” and explain its axes.
- Compare greedy decoding and beam search; do not promise beam search always improves answers.
- Try an unrelated question to demonstrate limitations honestly.
- Show a test prediction that failed, if present, and explain the next improvement.

## Final checks

- Extract the ZIP into a clean directory and launch with README commands.
- Run the unit tests and review notebook outputs.
- Ensure checkpoints, vocabulary, and configuration come from the same training run.
- Do not replace real training plots with invented results.
- Keep the model's LSTM and attention core intact when adding features.
- Record your own demonstration and be prepared to explain each file.

## Next research improvement

Collect diverse, reviewed support examples and design a harder test split (by conversation or source). Compare the same model with and without attention using the same split and training budget. Genuine multi-turn context requires a new input representation and conversation-level training examples; displaying message history alone is not model memory.
