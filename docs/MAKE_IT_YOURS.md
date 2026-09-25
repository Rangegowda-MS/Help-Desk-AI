# Make the project your own

A clear explanation of your choices is more useful in a project review than complicated code. This guide helps you understand and adapt the project; it does not guarantee any judgement about authorship. Follow your course's rules about acknowledging assistance.

## Safe personal changes

1. Change the heading and introduction in `draw_header()` in `app.py`.
2. Change `--accent` and the background colours in `assets/style.css`. Keep readable contrast. The matching Streamlit control colours are in `.streamlit/config.toml`.
3. Edit `SUGGESTIONS` in `app.py` to use questions relevant to your demonstration.
4. Add your own reviewed question/answer pairs to `data/support_faq.csv`. Do not include private customer details.
5. Retrain into a new folder, compare real results, and write down what improved or failed.

Changing the CSV does not change a trained model automatically. Retraining is required. Test a new run before replacing the `artifacts` folder used by the app.

## Understand the interface

| Function | Job |
|---|---|
| `setup_page` | Set the page options, load the CSS, and initialize session history |
| `load_bot` | Load the saved model once; reload when the checkpoint changes |
| `draw_sidebar` | Show decoding controls and the new-conversation button |
| `draw_header` | Show the introductory card |
| `generate_reply` | Call neural inference, record timing, and save the conversation turn |
| `draw_chat` | Handle suggested questions, typed questions, history, and export |
| `draw_reply` | Display the answer and input-limit information |
| `draw_attention` | Plot decoder attention over input tokens |
| `draw_results` | Read actual metrics, plot losses, and filter evaluation errors |
| `draw_project_notes` | Explain the model and locate its source files |
| `main` | Put the page together |

## Five things to practise explaining

- Why a word is represented by an integer ID and then an embedding.
- Why the encoder and decoder both use an LSTM.
- Why padding must be excluded from attention and loss.
- Why teacher forcing is available in training but not normal inference.
- Why a good loss value does not guarantee a correct response.

Try changing one setting, predict the effect, and check what actually happens. For example, compare greedy decoding with beam search on the same ten questions. Record the real outputs; do not assume one method always wins.

## Personal experiment log

Fill in these entries after doing your own experiments:

- Date and goal:
- Dataset changes:
- Training command:
- Best validation epoch and loss:
- Held-out answers that improved:
- Held-out answers that failed:
- What I learned:
- My next change:

## A practical preparation sequence

1. Run the app and try the four suggested questions.
2. Open the notebook and inspect how tokens are created.
3. Trace `Chatbot.reply()` from SOS to EOS.
4. Read a failed test prediction and explain the limitation.
5. Make one small change yourself and record the outcome.
6. Use the included viva guide to check your understanding.
