# Assignment 5: Self-Attention, Transformers, and Pretraining

**Course:** XCS224N — Natural Language Processing with Deep Learning (Stanford, Spring 2026)

## Overview

This assignment builds a character-level Transformer from a forked minGPT codebase, pretrains it on Wikipedia text using a simplified span-corruption objective (T5-style), and finetunes it to answer questions like "Where was [person] born?" The dramatic accuracy lift from pretraining (1.2% with no pretraining → 19.0% vanilla → 33.6% with RoPE) is the central empirical lesson — pretraining is what gives a small Transformer access to world knowledge it could never learn from a labeled finetuning set alone.

## Topics Covered

- Self-attention as parallel content-based lookup (Q/K/V mechanism, softmax-weighted value aggregation)
- Multi-headed attention and why it overcomes the fragility of single-head averaging under key perturbations
- The Transformer block: attention + position-wise feed-forward, with residuals and layer normalization
- Permutation equivariance of self-attention and why position embeddings are required for processing text
- Sinusoidal position embeddings and the irrationality argument for their uniqueness
- Rotary Position Embeddings (RoPE) implemented via complex-number multiplication
- Span corruption as a self-supervised pretraining objective
- The pretrain-then-finetune paradigm and parametric memory for knowledge access

## Key Files

- `src/submission/dataset.py` — Span corruption dataset for pretraining
- `src/submission/attention.py` — Causal self-attention with RoPE support
- `src/submission/helper.py` — Model initialization, pretrain, finetune, and train loops
- `tex/submission.tex` — Written responses (LaTeX)

## Related Course Modules

Modules 7–8: Self-Attention and Transformers, Pretraining
