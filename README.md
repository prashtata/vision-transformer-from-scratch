# Vision Transformer from Scratch

This project implements **Multi-Head Attention** from scratch in PyTorch, with an implementation in Vision Transformer classification model inspired by the Transformer architecture from "Attention is All You Need" and "An Image is Worth 16x16 Words." The goal is to understand and build attention mechanisms from the ground up, without relying on `torch.nn.MultiheadAttention`.

We are also laying the foundation for a **Vision Transformer (ViT) Autoencoder** trained on **CIFAR-10**, making this a modular and extensible project for modern deep learning education.
## What's Inside

```
multihead-attention-from-scratch/
├── attention/              # Custom MultiHeadAttention and scaled dot-product
├── vit/                    # Vision Transformer encoder, decoder, autoencoder
├── data/                   # CIFAR-10 dataset utilities
├── scripts/                # CLI-based training / toy examples
├── notebook/               # Interactive demo with visualizations
├── tests/                  # Unit tests for core components
├── README.md
├── requirements.txt
└── setup.py
```

## Modules Overview

### Multi-Head Attention (from scratch)

- Implements scaled dot-product attention
- Projects Q, K, V separately for each head
- Recombines with final linear projection
- Includes dropout and masking support

```python
from attention import MultiHeadAttention
x = torch.randn(2, 10, 64)
mha = MultiHeadAttention(d_model=64, num_heads=8)
out = mha(x, x, x)
```

### Vision Transformer (WIP)

- Modular encoder & decoder classes
- Full ViT-style autoencoder planned
- Clean training loop using CIFAR-10
- Will support patch embedding and masked autoencoding

## Unit Tests

Run all tests:

```bash
pytest tests/tests_mha.py
```

## Dependencies

```bash
git clone https://github.com/yourusername/multihead-attention-from-scratch.git
cd multihead-attention-from-scratch
pip install -r requirements.txt
```

## Demo Notebook

The `notebook/demo.ipynb` contains:

- Visualization of attention weights
- Explanation of shapes and internals
- Example inputs/outputs

## Dataset

We use **CIFAR-10** as the default image dataset for ViT experiments. You can fetch it easily via:

```python
from data import get_cifar10_dataloaders
train_loader, val_loader = get_cifar10_dataloaders(batch_size=128)
```

## Goals & Roadmap

- Multihead Attention from scratch
- Unit tests for attention module
- CIFAR-10 loader
- ViT encoder / decoder / autoencoder
- MAE training on CIFAR-10
- Positional encodings, masking logic
- Transformer block using our MHA
- Visualization of patch reconstruction loss

## Contributing

Feel free to fork the repo, open issues, or submit PRs

## Author

**Prasanth Tata**  
Machine Learning Engineer  
[GitHub](https://github.com/prashtata) • [LinkedIn](https://linkedin.com/in/prasanth-tata)

## License

MIT License — open to academic and research use. For commercial use, please reach out.
