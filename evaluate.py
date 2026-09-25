"""Generate held-out answers and exact-match rate, separate from teacher-forced loss."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from helpdesk.inference import Chatbot

def evaluate(directory='artifacts', beam_width=1):
    torch.set_num_threads(2)
    bot = Chatbot(directory)
    rows = []
    for item in pd.read_csv(Path(directory) / 'test.csv').itertuples():
        result = bot.reply(item.question, beam_width)
        rows.append(dict(question=item.question, reference=item.answer, generated=result['text'],
                         exact_match=result['text'] == item.answer, ended=result['ended']))
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(directory) / 'predictions.csv', index=False)
    summary = dict(exact_match=float(frame.exact_match.mean()),
                   eos_rate=float(frame.ended.mean()), examples=len(frame), beam_width=beam_width,
                   note='Exact match is strict; this tiny synthetic split does not measure real support quality.')
    (Path(directory) / 'generation_metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='artifacts')
    parser.add_argument('--beam-width', type=int, default=1, choices=[1, 3, 5])
    evaluate(**vars(parser.parse_args()))
