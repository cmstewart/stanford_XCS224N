# Assignment 3: Dependency Parsing

**Course:** XCS224N - Natural Language Processing with Deep Learning (Stanford, Spring 2026)

## Overview

This assignment focuses on dependency parsing, the task of analyzing the grammatical structure of a sentence by identifying relationships between words. It involves implementing a transition-based dependency parser using a neural network, combining concepts from earlier modules (embeddings, feedforward networks) with structured prediction.

## Topics Covered

- Dependency grammar and parse tree structure
- Transition-based parsing with shift-reduce operations (SHIFT, LEFT-ARC, RIGHT-ARC)
- Implementing the transition mechanics (stack, buffer, and dependency tracking)
- Building a neural network classifier (using PyTorch) to predict parser transitions
- Training the parser on real dependency treebank data
- Evaluating parser accuracy (UAS - Unlabeled Attachment Score)

## Key Files

- `src/submission/parser_transitions.py` - Implementation of shift-reduce transition operations
- `src/submission/parser_model.py` - Neural network model for predicting transitions
- `src/submission/train.py` - Training loop for the parser

## Related Course Modules

Modules 3-4: Neural Networks, Backpropagation, and Dependency Parsing
