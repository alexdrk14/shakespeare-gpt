# Shakespeare GPT Model

A PyTorch implementation of a GPT-style language model trained on Shakespeare's works. This model generates Shakespeare-like text using a transformer-based architecture.

## Overview

This project implements a compact version of the GPT (Generative Pre-trained Transformer) architecture trained on Shakespeare's texts. The model learns to predict the next character in a sequence, allowing it to generate Shakespeare-style text.

## Features

- Character-level language modeling
- Multi-head self-attention mechanism
- Position embeddings
- Layer normalization
- Configurable architecture parameters

## Model Architecture

- Token Embeddings: 384-dimensional embeddings for each character
- Position Embeddings: Learnable position encodings
- Transformer Blocks: 6 layers of self-attention and feed-forward networks
- Multi-head Attention: 6 attention heads
- Layer Normalization: Applied before each sub-block
- Dropout: 0.2 for regularization

## Hyperparameters

```python
batch_size = 64        # Number of sequences processed in parallel
block_size = 256       # Maximum context length
n_embed = 384         # Embedding dimension
n_layers = 6         # Number of transformer blocks
n_heads = 6          # Number of attention heads
dropout = 0.2        # Dropout rate
learning_rate = 3e-4  # Learning rate for AdamW optimizer
```

## Training

The model is trained with the following specifications:
- Training/Validation split: 90/10
- Optimizer: AdamW
- Maximum iterations: 5000
- Evaluation interval: Every 500 iterations
- Loss: Cross-entropy loss
- Device: CUDA (GPU) if available, otherwise CPU

## Usage

1. The script will automatically download the Shakespeare dataset if not present:

```python
if not os.path.exists('input.txt'):
    _ = wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
```

2. To train a new model:
```python
model = BigramLanguageModel(vocab_size)
model = model.to(device)
training_loop(model)
```

3. To generate text:
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
```

## Model Components

### 1. Head Class
- Implements single head of self-attention
- Includes Query, Key, and Value projections
- Masked attention for autoregressive generation

### 2. MultiHeadAttention Class
- Parallel processing of multiple attention heads
- Concatenation and projection of head outputs

### 3. FeedForward Class
- Two-layer neural network with ReLU activation
- Expansion ratio of 4x in hidden layer

### 4. Block Class
- Combines self-attention and feed-forward layers
- Includes layer normalization and residual connections

### 5. CharacterBasedGPT Class
- Main model class combining all components
- Handles token and position embeddings
- Implements text generation logic

## Requirements

- PyTorch
- wget (for downloading dataset)
- CUDA-capable GPU (optional but recommended)

## Model Persistence

The model automatically saves after training:
```python
torch.save(model, 'shakespeare_gpt_model.pt')
```

And can be loaded later:
```python
model = torch.load('shakespeare_gpt_model.pt', weights_only=False)
```

## Acknowledgments

This implementation is inspired by [Andrej Karpathy](https://github.com/karpathy) and his set of [AI open-courses](https://karpathy.ai/). 
## License

This project is open-source and available under the MIT License.
