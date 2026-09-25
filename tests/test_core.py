import unittest
import torch
from helpdesk.data import clean, Vocabulary, PAD, SOS, EOS, collate, split_pairs, load_pairs
from helpdesk.model import Seq2Seq

class CoreTests(unittest.TestCase):
    def test_clean_and_specials(self):
        self.assertEqual(clean(' HELLO!!! 😊 world ', 1), 'hello')
        self.assertEqual(Vocabulary.build(['hello']).encode('missing'), [3, EOS])

    def test_no_prompt_leakage(self):
        parts = split_pairs(load_pairs('data/support_faq.csv'))
        sets = [set(p.question) for p in parts]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_padding_attention_and_gradients(self):
        torch.manual_seed(1)
        model = Seq2Seq(12, 8, 16, 0.)
        source, lengths, target = collate([(torch.tensor([4,5,EOS]), torch.tensor([6,EOS])),
                                          (torch.tensor([7,EOS]), torch.tensor([8,9,EOS]))])
        enc, state = model.encoder(source, lengths)
        _, weights = model.decoder.attention(state[0][-1], enc, source.ne(PAD))
        self.assertEqual(float(weights[1,-1].detach()), 0.)
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2)))
        logits = model(source, lengths, target, 1.)
        self.assertEqual(tuple(logits.shape), (2,3,12))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target.flatten(), ignore_index=PAD)
        loss.backward()
        self.assertTrue(torch.isfinite(model.encoder.lstm.weight_ih_l0.grad).all())
        # Packed encoder state is unaffected by appended padding.
        extra = torch.cat([source, torch.zeros((2,2), dtype=torch.long)], 1)
        _, other = model.encoder(extra, lengths)
        self.assertTrue(torch.allclose(state[0], other[0], atol=1e-6))

if __name__ == '__main__': unittest.main()
