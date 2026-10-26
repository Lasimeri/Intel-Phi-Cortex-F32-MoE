# Stage 6 (interim): chat fine-tune on oasst1 guanaco with the Human/Assistant turn format

Not committed: these results and the data pipeline (tools/jsonl-text.c, data/oasst/) stay out of the repository by the
user's instruction. Model, vocabulary (TinyStories BPE 4096) and training code unchanged; only the data changed.

## Data

- openassistant-guanaco (HF timdettmers/openassistant-guanaco), train 4,221 docs / 7,082,125 B after the marker rewrite
  ("### Human: " -> "\n\nHuman: ", "### Assistant: " -> "\n\nAssistant: ", "<|endoftext|>" between docs; docs with more
  than 2 % non-ASCII bytes dropped); valid 213 docs. sha256 of the source files in data/oasst/sha256.txt.
- Encoded with the TinyStories vocabulary: train 2,783,587 tokens (2.544 B/token), valid 143,542 tokens; round trip
  byte-identical.

## Run (card /tmp/cortex/chat1, log results/s06-chat-finetune.log)

    CX_INIT=/tmp/cortex/run4off/ckpt.bin CX_BUS=0 CX_CKPT_EVERY=20000 ./cluster-gate run chat-train.u16 chat1 32 2783000 1e-3 chat-valid.u16 143500 500

| quantity | value |
|---|---|
| start | TinyStories cluster run4off (1M tokens, bus off) |
| tokens | 2,783,000 (one epoch), L 32, lr 1e-3 constant, Adam 0.9/0.95 |
| rate | 1,211 to 1,222 tok/s (fwd+vote+head 325 to 326 us, bwd+grads+adam 495 to 497 us per step) |
| wall | train 2,295.3 s, valid 45.8 s |
| train loss | 7.71 bits/token at 320k, 7.15 to 7.57 from 1.6M on (plateau) |
| valid (143,488 tokens) | 5.3487 nats = 7.7166 bits/token = 1.9481 bits/byte |
| baseline (same cluster before the fine-tune, same valid set) | 11.9901 bits/token = 3.0270 bits/byte (results/s06-chat-baseline.log) |

## Generation (results/s06-chat-demo.log, cortexd + host cortex, --chat --bus 0)

Sampled (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2, seed 1), 120 tokens, prompt "What is the capital of France?":

    Assistant: The dataction interferience, a dett may be interformations. Theproritions to the (relebles can be
    atutchnce or mainerenties of thestimumansicality that is an example and their red you use.

    Assistant: The iveion, it is used to be Pythonproperience for a uscationing, and the audatural dat

"How do I make a cup of tea?" stopped itself after 71 tokens by emitting the "Human" turn marker. Rate 1,505 to 1,521
tok/s overall (card 382 to 391 us per token, host sampler 327 to 341 us).

What the fine-tune did: the turn structure is learned (the model opens with "Assistant:", emits further turn markers,
stops on "Human"), the register moved from TinyStories to assistant prose (Python, interface, data, module, example,
system), and the validation cross-entropy fell from 11.99 to 7.72 bits/token. What it did not do: word-level coherence.
At 7.7 bits over a 4096-token vocabulary the next-token distribution is still broad (perplexity about 210), which is
what the sampled text shows: plausible subword shapes, few real words, no sentence meaning. The cluster is small
(120 x 44,512 + 524,288 embedding parameters, about 5.9M) and saw 2.8M chat tokens once.

## Open

- More epochs (concatenate the token file or add an epoch loop) and a lower learning rate for the second pass.
- The bus stays off: it hurt validation loss on TinyStories (s04) and was not retried here.
