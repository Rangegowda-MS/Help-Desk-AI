"""Streamlit interface. Start with: python -m streamlit run app.py"""
import json
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from helpdesk.inference import Chatbot

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
DECODERS = {'Greedy': 1, 'Beam search (3)': 3, 'Beam search (5)': 5}
SUGGESTIONS = {
    'Reset password': 'How do I reset my password?',
    'Find an invoice': 'Where is my invoice?',
    'Invite a teammate': 'How do I invite a teammate?',
    'Contact support': 'How can I reach support?',
}


def setup_page():
    st.set_page_config(page_title='HelpDesk AI', page_icon='💬', layout='wide')
    styles = (ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')
    st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)
    st.session_state.setdefault('messages', [])


@st.cache_resource
def load_bot(checkpoint_timestamp):
    # A changed checkpoint timestamp causes Streamlit to reload the model.
    torch.set_num_threads(2)
    return Chatbot(ARTIFACTS)


def read_json(filename):
    path = ARTIFACTS / filename
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def draw_sidebar(bot):
    with st.sidebar:
        st.markdown('''
        <div class="brand">
          <div class="brand-mark">h.</div>
          <div><div class="brand-name">HelpDesk AI</div>
          <div class="brand-subtitle">Your support workspace</div></div>
        </div>''', unsafe_allow_html=True)
        if st.button('＋ New conversation', key='new_chat', width='stretch', type='primary'):
            st.session_state.messages = []
        st.divider()
        st.caption('CONVERSATION SETTINGS')
        mode = st.selectbox('Response method', list(DECODERS), key='decoder')
        attention = st.toggle('Show attention explorer', value=True, key='attention')
        st.caption('Greedy chooses one word at a time. Beam search compares several possible replies.')
        st.divider()
        st.markdown('<div class="status"><span class="status-dot"></span>Model loaded</div>',
                    unsafe_allow_html=True)
        st.caption(f'{len(bot.vocab):,} vocabulary tokens · runs on your computer')
        with st.expander('About this demo'):
            st.write('Trained on a small set of synthetic support questions. Replies may be inaccurate.')
            st.write('The model uses your current question only. It cannot change accounts or perform actions.')
    return mode, attention


def draw_header():
    st.markdown('''
    <div class="hero">
      <div class="eyebrow">A little help, one conversation at a time</div>
      <h1>Questions happen.<br><span>Let’s work through them.</span></h1>
      <p>Ask about your account, billing, or a technical issue.
      Explore how a small language model builds a reply, word by word.</p>
      <div class="tags"><span class="tag">Account help</span>
      <span class="tag">Billing questions</span><span class="tag">Troubleshooting</span></div>
    </div>''', unsafe_allow_html=True)


def draw_attention(result):
    with st.expander('Why these words? Explore attention'):
        st.caption('Each row is a generated word. Brighter cells mean more weight on that input word.')
        fig, ax = plt.subplots(figsize=(8, max(2.8, len(result['output_tokens']) * .32)))
        fig.patch.set_facecolor('#14232e')
        ax.set_facecolor('#14232e')
        matrix = np.asarray(result['attention'])
        plot = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(result['input_tokens'])), result['input_tokens'], rotation=40, ha='right')
        ax.set_yticks(range(len(result['output_tokens'])), result['output_tokens'])
        ax.set_xlabel('Question tokens', color='#bacbd6')
        ax.set_ylabel('Reply tokens', color='#bacbd6')
        ax.tick_params(colors='#d9e5ec', labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(plot, ax=ax, pad=.02)
        bar.ax.tick_params(colors='#d9e5ec')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption('Attention weights describe the model’s calculation; they do not establish correctness.')


def draw_reply(result, show_attention):
    # Keep the raw model output in exports; add only sentence casing for display.
    text = result['text']
    st.write(text[:1].upper() + text[1:])
    if result.get('unknown_ratio', 0) > .3:
        st.caption('Some words were not in the training vocabulary. Try a simpler question.')
    if result.get('truncated'):
        st.caption(f"This reply used the first {result['input_limit']} cleaned words of your question.")
    if result.get('attention'):
        ending = 'Reply complete' if result['ended'] else 'Length limit reached'
        st.caption(f"{result['latency_ms']:.0f} ms · {result['decoding']} · {ending}")
        if show_attention:
            draw_attention(result)


def generate_reply(bot, prompt, mode):
    # Generate first so a failed request does not leave an incomplete turn in history.
    with st.spinner('Writing a reply…'):
        start = time.perf_counter()
        result = bot.reply(prompt, DECODERS[mode])
        result['latency_ms'] = (time.perf_counter() - start) * 1000
    result['decoding'] = mode
    result['input_limit'] = bot.config['max_words']
    cleaned_words = re.sub(r'[^a-z0-9\s]', ' ', prompt.lower()).split()
    result['truncated'] = len(cleaned_words) > result['input_limit']
    st.session_state.messages.extend([
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'result': result},
    ])


def draw_chat(bot, mode, show_attention):
    st.markdown('<div class="section-heading">Your conversation</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-note">Start with a question below, or write your own.</p>',
                unsafe_allow_html=True)
    suggestion = None
    for column, (label, question) in zip(st.columns(4), SUGGESTIONS.items()):
        if column.button(label, key=f'ask_{label}', width='stretch'):
            suggestion = question

    if not st.session_state.messages and not suggestion:
        st.markdown('''<div class="empty-state"><div class="empty-symbol">✦</div>
        <h3>What can I help you with?</h3>
        <p>Try “How do I reset my password?” to get started.</p></div>''', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar='🟢' if message['role'] == 'assistant' else None):
            if message['role'] == 'user':
                st.write(message['content'])
            else:
                draw_reply(message['result'], show_attention)

    typed_question = st.chat_input('Write your support question…', max_chars=500)
    question = typed_question or suggestion
    if question:
        generate_reply(bot, question, mode)
        st.rerun()

    if st.session_state.messages:
        left, right = st.columns([3, 2])
        turns = len(st.session_state.messages) // 2
        left.caption(f'{turns} question' + ('s' if turns != 1 else '') + ' in this session')
        right.download_button(
            '↓ Save conversation', json.dumps(st.session_state.messages, indent=2),
            file_name='helpdesk-conversation.json', mime='application/json', width='stretch')
    st.caption('Demo responses · English · Up to 15 cleaned input words · Each question is handled separately')


def draw_results():
    st.subheader('The numbers behind the model')
    st.caption('Results from the saved training run. These values are read from the training artifacts.')
    metrics = read_json('metrics.json')
    if not metrics:
        st.info('Train the model to see its results here.')
        return
    generation = read_json('generation_metrics.json')
    columns = st.columns(4)
    columns[0].metric('Training examples', metrics['training_pairs'])
    columns[1].metric('Selected epoch', metrics['best_epoch'])
    columns[2].metric('Test perplexity', f"{metrics['test_perplexity']:.2f}")
    if generation:
        columns[3].metric('Exact answer match', f"{generation['exact_match']:.1%}")
    else:
        columns[3].metric('Test examples', metrics['test_pairs'])
    st.write('')
    history_path = ARTIFACTS / 'history.csv'
    if history_path.exists():
        st.markdown('**Learning over time**')
        history = pd.read_csv(history_path).set_index('epoch')
        st.line_chart(history[['train_loss', 'validation_loss']].rename(columns={
            'train_loss': 'Training loss', 'validation_loss': 'Validation loss'}),
            color=['#7ce2c4', '#e7b970'], height=290)
        st.caption('Cross-entropy per token. Training uses scheduled teacher forcing; validation uses full teacher forcing.')
    predictions_path = ARTIFACTS / 'predictions.csv'
    if predictions_path.exists():
        st.markdown('**How did the held-out questions go?**')
        predictions = pd.read_csv(predictions_path)
        only_misses = st.checkbox('Show answers that did not exactly match', key='only_misses')
        visible = predictions[~predictions.exact_match] if only_misses else predictions
        st.dataframe(visible[['question', 'reference', 'generated', 'exact_match']],
                     hide_index=True, width='stretch')
        st.download_button('Download evaluation CSV', predictions_path.read_bytes(),
                           'helpdesk-evaluation.csv', 'text/csv')
    st.info('Small synthetic dataset: related paraphrases appear across splits. These scores do not measure real-world support reliability.')


def draw_project_notes():
    st.subheader('From a question to a reply')
    st.caption('Three parts work together. You can inspect each one in the source code.')
    steps = [
        ('01 / READ', 'The encoder', 'Words become vectors. An LSTM reads the question and keeps a representation of each word.'),
        ('02 / FOCUS', 'The attention layer', 'The decoder assigns weights to input positions and combines them into a context vector.'),
        ('03 / WRITE', 'The decoder', 'A second LSTM predicts one token at a time, using its previous output until it reaches the end token.'),
    ]
    for column, (number, title, body) in zip(st.columns(3), steps):
        column.markdown(f'<div class="step-card"><div class="step-number">{number}</div>'
                        f'<h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.write('')
    with st.expander('What happens during training?'):
        st.write('The model predicts answer tokens and receives a cross-entropy loss. Teacher forcing supplies correct previous tokens for part of training. Backpropagation updates the weights.')
        st.write('Dropout and gradient clipping help training. Validation loss selects the saved checkpoint, and early stopping avoids unnecessary epochs.')
    with st.expander('Where is each part of the code?'):
        st.table(pd.DataFrame([
            ['helpdesk/data.py', 'Cleaning, vocabulary, and padded batches'],
            ['helpdesk/model.py', 'Encoder, attention, and decoder'],
            ['train.py', 'Training and checkpoint exports'],
            ['helpdesk/inference.py', 'Greedy decoding and beam search'],
            ['app.py', 'Chat interface and results dashboard'],
            ['assets/style.css', 'Colours, spacing, and responsive layout'],
        ], columns=['File', 'Purpose']))
    with st.expander('What are the current limitations?'):
        st.write('The dataset contains synthetic FAQs for a fictional service. The model has a limited vocabulary and can produce incorrect replies.')
        st.write('Visible chat history does not provide model memory. Only the current question is encoded. The app does not connect to real accounts or perform support actions.')
    st.caption('For the full walkthrough, open notebooks/HelpDesk_AI.ipynb. For presentation practice, read docs/EXPLANATION_AND_VIVA.md.')


def main():
    setup_page()
    checkpoint = ARTIFACTS / 'best.pt'
    if not checkpoint.exists():
        st.title('HelpDesk AI')
        st.warning('A trained model is needed before you can start chatting.')
        st.code('python train.py --epochs 50')
        st.stop()
    bot = load_bot(checkpoint.stat().st_mtime_ns)
    mode, show_attention = draw_sidebar(bot)
    draw_header()
    chat, results, notes = st.tabs(['Conversation', 'Training results', 'Project notes'])
    with chat:
        draw_chat(bot, mode, show_attention)
    with results:
        draw_results()
    with notes:
        draw_project_notes()
    st.markdown('<div class="footer-note">HELPDESK AI / A SEQ2SEQ LEARNING PROJECT</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
