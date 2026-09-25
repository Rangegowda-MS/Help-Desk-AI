# Beginner explanation and viva preparation

## What happens when you type a question?

Suppose you type "reset my password". Cleaning makes the text lowercase and removes unwanted symbols. A vocabulary converts each word to an integer. An embedding turns each integer into a learned numerical vector. The encoder LSTM reads those vectors and produces a hidden state for each input position, plus final hidden and cell states.

The decoder receives the final encoder states and a start token. Attention compares the decoder hidden state with every encoder output. A softmax turns the resulting scores into weights that sum to one. Their weighted sum is a context vector. The decoder uses the context and previous token to predict the next token. The predicted token becomes the next input. This repeats until the end token, with a length limit as a safeguard.

## Important terms

| Term | Meaning here |
|---|---|
| Seq2Seq | Convert one variable-length sequence into another |
| LSTM | Recurrent network with gates and a cell state to retain information |
| Embedding | A trainable vector associated with each vocabulary token |
| SOS | Token that starts decoding |
| EOS | Token indicating the answer has ended |
| PAD | Token used to make batch sequences the same length |
| UNK | Token for a word absent from the training vocabulary |
| Attention | A weighted combination of input-position representations |
| Teacher forcing | Feed the correct previous answer token during training |
| Cross-entropy | Loss penalizing low probability assigned to the target token |
| Perplexity | exp(mean target-token negative log probability); lower is better on the same data/vocabulary |
| Epoch | One pass through the training set |
| Beam search | Keep several candidate output sequences during decoding |

## Questions and answers

**1. Why not use only a fixed context vector?**
It may bottleneck information from longer input sequences. Attention gives the decoder access to all encoder outputs at each step. This benefit should be measured with a controlled experiment rather than assumed from a plot alone.

**2. What attention mechanism is implemented?**
Bahdanau additive attention: score each encoder output against the decoder hidden state through learned linear projections, tanh, and a scalar projection. Mask padded positions, apply softmax, then take the weighted sum of encoder outputs.

**3. Does CrossEntropyLoss need one-hot labels?**
No. PyTorch accepts integer class labels and raw logits. It combines log-softmax with negative log likelihood. The labels are the next-token IDs.

**4. Why ignore padding?**
Padding is an artificial batching convenience. Including it in the loss would reward learning filler tokens. Attention must also avoid attending to padding.

**5. What is different between training and inference?**
Training can use correct previous tokens and computes gradients. Inference has no target answer, so it feeds its predictions back without gradients.

**6. Why schedule teacher forcing?**
The ratio gradually falls from 1.0 toward 0.5 so training sometimes exposes the decoder to its own mistakes. Validation perplexity uses full teacher forcing for a consistent conditional-likelihood metric.

**7. What is gradient clipping?**
It limits the overall gradient norm to reduce exploding-gradient problems in recurrent training. This project uses a maximum norm of 1.0.

**8. Why split the data before creating the vocabulary?**
To avoid using validation and test text to define the training vocabulary. Unseen words become UNK.

**9. What counts as context here?**
The words of the current question, represented by the encoder and attention. Previous chat messages are displayed but are not passed to the network.

**10. What is stored in the model?**
Learned weight tensors. The combined checkpoint also stores vocabulary, architecture parameters, configuration, selected epoch, and validation loss. It does not call an external language model.

**11. What do the attention heatmap axes mean?**
Columns are input tokens; rows are generated tokens, including EOS if reached. Each row contains weights over input positions.

**12. How do you evaluate this project?**
Use held-out token loss/perplexity, free-running generated answers, exact match, EOS completion rate, and manual response review. Exact match can reject a valid alternative answer. The tiny synthetic split is not evidence of production reliability.

**13. Why may a fluent answer be wrong?**
The model optimizes token prediction, not factual verification. A small dataset may encourage memorized or generic answers. Attention weights do not validate the answer.

**14. What are your advanced additions?**
Beam search, attention inspection, scheduled teacher forcing, clipping, early stopping, metrics dashboard, export, and automated core checks.

**15. What would you improve next?**
Dataset diversity, separate evaluation by source or conversation, a no-attention baseline, and a genuine multi-turn training setup. Changes should be measured on a fixed test set.
