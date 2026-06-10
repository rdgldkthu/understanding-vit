# understanding-vit

A step-by-step implementation of the Vision Transformer (ViT) from scratch using PyTorch,
following *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2020).

The goal is **deep understanding**, not fast results. Each component is built by hand
with shape annotations and explanations before moving to the next stage.

---

## Stages

| Notebook | Topic | What you build |
| --- | --- | --- |
| `01_attention.ipynb` | Scaled Dot-Product Attention | `scaled_dot_product_attention`, `SelfAttention`, `MultiHeadAttention` |
| `02_mini_transformer.ipynb` | Mini Transformer | `TransformerBlock`, `MiniTransformer` — trained to ~89% on MNIST |
| `03_vit_cifar10.ipynb` | Vision Transformer | `PatchEmbedding`, `ViT` — trained to ~64% on CIFAR-10 |
| `04_visualization.ipynb` | Attention Rollout | Heatmaps showing which patches the model attends to |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision einops matplotlib seaborn timm jupyter

# verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# launch notebooks
jupyter notebook --no-browser --port=8888
```

---

## Results

| Model | Dataset | Accuracy | Notes |
| --- | --- | --- | --- |
| MiniTransformer | MNIST | ~89% | Raw pixel rows as tokens, no positional encoding |
| ViT (patch=4) | CIFAR-10 | ~64% | Trained from scratch, 50 epochs |

ViTs are data-hungry and lack the spatial inductive bias of CNNs. The accuracy gap vs. CNNs
(~93% on CIFAR-10) reflects the cost of learning spatial structure entirely from data.

---

## Reference

- Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- PyTorch docs: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- Pretrained ViTs (timm): [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
