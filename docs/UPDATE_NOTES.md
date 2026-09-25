# Interface update

## What changed

- Navy background, mint accents, a clearer welcome panel, and consistent cards.
- Clickable example questions that use the same trained model as typed questions.
- Response timing, sentence casing, and an accurate long-input notice.
- A styled attention explorer with labelled input and output tokens.
- An interactive loss chart and a filter for failed exact-reference matches.
- Conversation and evaluation CSV downloads.
- Separate CSS file and small named interface functions.
- Readability cleanup of compressed statements in the training script.
- Personalization guide and experiment log template.

## Model and results

The architecture, dataset, model weights, and reported scores remain from the original training run. This is an interface and readability update, not a claim of improved model accuracy. No authorship or detection guarantees are made.

## Updating on Windows

1. Stop the running app with Ctrl+C in its terminal.
2. Back up any files you edited or trained yourself.
3. Extract the updated ZIP and copy its HelpDesk-AI contents into your existing project folder.
4. Keep your existing `.venv` folder; it is not included in the ZIP.
5. Be sure to copy `assets/style.css` and `.streamlit/config.toml` as well as the Python files.
6. From the project folder, launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies are still missing:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code Python interpreter to make import diagnostics use the same installed packages.

## Verification of this update

All four automated tests passed. The interface check covered launching the app, clicking an example question, typing a message, selecting beam search, toggling attention, filtering evaluation failures, handling symbol-only input, showing the long-input notice, and resetting the conversation. The original model checks also passed. Tests ran on Linux; the Windows launch instructions are supplied for your local setup.
