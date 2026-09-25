# Vendored dependencies

`LitePT/` contains the minimal source needed by the point-cloud models. It is
based on [`prs-eth/LitePT`](https://github.com/prs-eth/LitePT) commit
`436d04801c8151faebe66a1b2d368a9711e7e6aa` (BSD-3-Clause).

The local `litept/model.py` adds two runtime compatibility changes used by
this project:

- a PyTorch scaled-dot-product-attention fallback when `flash-attn` is not
  installed;
- `torch.segment_reduce` in place of the optional `torch-scatter` package.

The original LitePT license is kept in `LitePT/LICENSE`. The complete upstream
research checkout can be restored independently from the URL and revision
above; it is not required for this project's training or inference scripts.

`ForAINet/` is an unmodified local research checkout and is intentionally not
versioned because no executable in this repository imports it.
