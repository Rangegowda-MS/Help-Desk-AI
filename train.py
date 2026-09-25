"""Run from project root: python train.py --epochs 50."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from helpdesk.data import PAD, Vocabulary, load_pairs, split_pairs, Pairs, collate
from helpdesk.model import Seq2Seq


def token_loss(model, loader, device):
    model.eval()
    loss_sum, tokens = 0., 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
    with torch.inference_mode():
        for source, lengths, target in loader:
            source, target = source.to(device), target.to(device)
            logits = model(source, lengths, target, teacher_forcing=1.)
            loss_sum += criterion(logits.flatten(0, 1), target.flatten()).item()
            tokens += target.ne(PAD).sum().item()
    return loss_sum / tokens


def train(data='data/support_faq.csv', output='artifacts', epochs=50, batch_size=32,
          embedding_dim=64, hidden_dim=128, lr=.003, dropout=.15, seed=42,
          max_words=15, patience=12, device='cpu'):
    if epochs < 1 or max_words < 1 or batch_size < 1:
        raise ValueError('epochs, max_words, and batch_size must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_pairs(data, max_words)
    train_df, val_df, test_df = split_pairs(frame, seed)
    vocab = Vocabulary.build(list(train_df.question) + list(train_df.answer))
    loaders = []
    for name, part in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        part.to_csv(out / f'{name}.csv', index=False)
        loaders.append(DataLoader(Pairs(part, vocab), batch_size=batch_size,
                                  shuffle=(name == 'train'), collate_fn=collate))
    architecture = dict(embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
    config = dict(max_words=max_words, seed=seed, epochs_requested=epochs, batch_size=batch_size,
                  learning_rate=lr, patience=patience, data_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest())
    model = Seq2Seq(len(vocab), **architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
    best, bad_epochs, history = float('inf'), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0., 0
        ratio = max(.5, 1 - .5 * (epoch - 1) / max(epochs - 1, 1))
        for source, lengths, target in loaders[0]:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(source, lengths, target, ratio)
            loss = criterion(logits.flatten(0, 1), target.flatten())
            n = target.ne(PAD).sum()
            (loss / n).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item()
            count += n.item()
        val_loss = token_loss(model, loaders[1], device)
        history.append(dict(epoch=epoch, train_loss=total_loss/count, validation_loss=val_loss,
                            validation_perplexity=math.exp(min(val_loss, 30)), teacher_forcing=ratio))
        if val_loss < best - .0001:
            best, bad_epochs = val_loss, 0
            torch.save(dict(state_dict=model.state_dict(), vocabulary=vocab.words,
                            architecture=architecture, config=config, epoch=epoch,
                            validation_loss=val_loss), out / 'best.pt')
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: train={total_loss/count:.4f} validation={val_loss:.4f}', flush=True)
        if bad_epochs >= patience:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break
    checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    torch.save(model.encoder.state_dict(), out / 'encoder.pt')
    torch.save(model.decoder.state_dict(), out / 'decoder.pt')
    (out / 'vocabulary.json').write_text(json.dumps(vocab.words, indent=2), encoding='utf-8')
    (out / 'config.json').write_text(json.dumps(dict(config=config, architecture=architecture), indent=2))
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    test_loss = token_loss(model, loaders[2], device)
    report = dict(best_epoch=checkpoint['epoch'], epochs_completed=len(history),
                  training_pairs=len(train_df), validation_pairs=len(val_df), test_pairs=len(test_df),
                  vocabulary_size=len(vocab), validation_loss=best,
                  test_loss=test_loss, test_perplexity=math.exp(min(test_loss, 30)),
                  note='Synthetic FAQ demo; random prompt split includes related paraphrases. Not a production benchmark.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Train (scheduled teacher forcing)')
    ax.plot([h['epoch'] for h in history], [h['validation_loss'] for h in history], label='Validation (full teacher forcing)')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy per non-padding token', title='HelpDesk AI | Actual training run')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'loss.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/support_faq.csv')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--dropout', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-words', type=int, default=15)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    train(**vars(parser.parse_args()))
