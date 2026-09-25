"""Greedy and length-normalized beam search. No FAQ retrieval or API replies."""
import math
from pathlib import Path
import torch
from .data import Vocabulary, clean, PAD, SOS, EOS, UNK
from .model import Seq2Seq

class Chatbot:
    def __init__(self, directory='artifacts', device='cpu'):
        self.device = torch.device(device)
        self.bundle = torch.load(Path(directory) / 'best.pt', map_location=self.device, weights_only=True)
        self.vocab = Vocabulary(self.bundle['vocabulary'])
        self.config = self.bundle['config']
        self.model = Seq2Seq(len(self.vocab), **self.bundle['architecture']).to(self.device)
        self.model.load_state_dict(self.bundle['state_dict'])
        self.model.eval()

    @torch.inference_mode()
    def reply(self, text, beam_width=1):
        if beam_width not in (1, 3, 5):
            raise ValueError('beam_width must be 1, 3, or 5')
        normalized = clean(text, self.config['max_words'])
        if not normalized:
            return {'text': 'Please enter a question using letters or numbers.', 'attention': [],
                    'input_tokens': [], 'output_tokens': [], 'unknown_ratio': 0.0, 'ended': False}
        ids = self.vocab.encode(normalized)
        source = torch.tensor([ids], device=self.device)
        outputs, state = self.model.encoder(source, torch.tensor([len(ids)]))
        mask = source.ne(PAD)
        # Each candidate owns token history, recurrent state, log probability, attention.
        beams = [([SOS], state, 0., [])]
        rank = lambda b: b[2] / (((5 + len(b[0]) - 1) / 6) ** .6)
        for _ in range(self.config['max_words'] + 1):
            candidates = []
            for tokens, previous, score, attention in beams:
                if tokens[-1] == EOS:
                    candidates.append((tokens, previous, score, attention))
                    continue
                logits, next_state, weights = self.model.decoder(
                    torch.tensor([tokens[-1]], device=self.device), previous, outputs, mask)
                logits[0, [PAD, SOS, UNK]] = float('-inf')
                values, indices = logits.log_softmax(-1).topk(beam_width, dim=-1)
                for value, index in zip(values[0].tolist(), indices[0].tolist()):
                    candidates.append((tokens + [index], next_state, score + value,
                                       attention + [weights[0].cpu().tolist()]))
            beams = sorted(candidates, key=rank, reverse=True)[:beam_width]
            if all(b[0][-1] == EOS for b in beams):
                break
        tokens, _, score, attention = max(beams, key=rank)
        output = [self.vocab.words[t] for t in tokens[1:]]
        words = [w for w in output if w != '<EOS>']
        return {'text': ' '.join(words) or 'No response generated. Please rephrase.',
                'attention': attention, 'input_tokens': [self.vocab.words[i] for i in ids],
                'output_tokens': output, 'unknown_ratio': ids[:-1].count(UNK) / max(1, len(ids)-1),
                'ended': tokens[-1] == EOS,
                'token_score': math.exp(score / max(1, len(tokens)-1))}
