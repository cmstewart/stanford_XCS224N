# Assignment 4: Neural Machine Translation with RNNs

**Course:** XCS224N - Natural Language Processing with Deep Learning (Stanford, Spring 2026)

## Overview

This assignment implements a sequence-to-sequence neural machine translation (NMT) system for translating Cherokee to English. The model uses a bidirectional LSTM encoder, a unidirectional LSTM decoder, and global attention (Luong et al., 2015). The assignment covers the full pipeline from padding and embedding inputs, through encoding and decoding, to training on a GPU and evaluating with BLEU scores.

## Topics Covered

- Sequence-to-sequence architecture with encoder-decoder framework
- Bidirectional LSTMs for encoding variable-length source sentences
- Attention mechanism: computing alignment scores, attention weights, and context vectors
- Packed/padded sequences for efficient batched processing of variable-length inputs
- LSTM hidden state vs. cell state and their roles in the network
- Sentence padding and embedding layers with padding indices
- Projection layers for bridging encoder and decoder dimensions
- BLEU score computation and its properties as an evaluation metric
- Training on Azure VMs with GPU acceleration
- Beam search decoding for generating translations

## Key Files

- `src/submission/utils.py` - Sentence padding utility
- `src/submission/model_embeddings.py` - Source and target embedding layers
- `src/submission/nmt_model.py` - Full NMT model (encoder, decoder, attention, step)
- `src/run.sh` - Scripts for vocab generation, training, and testing
- `src/chr_en_data/` - Cherokee-English parallel training and test data

## Results

- Corpus BLEU score: 12.13 (Cherokee to English translation)

## Related Course Modules

Modules 5-6: Language Models, RNNs, and Machine Translation
