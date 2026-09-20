import torch


def precompute_rope(
    max_seq_len,
    head_dim,
    theta=10000.0
):

    device = torch.device("cpu")

    positions = torch.arange(
        max_seq_len,
        device=device
    ).float()

    frequencies = 1.0 / (
        theta ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device
            ).float()
            / head_dim
        )
    )

    angles = torch.outer(
        positions,
        frequencies
    )

    cos = torch.cos(angles)

    sin = torch.sin(angles)

    return cos, sin


def rotate_half(x):

    x1 = x[..., ::2]

    x2 = x[..., 1::2]

    return torch.stack(
        (-x2, x1),
        dim=-1
    ).flatten(-2)


def apply_rope(
    x,
    cos,
    sin
):

    head_dim = x.shape[-1]

    cos = torch.repeat_interleave(
        cos,
        2,
        dim=-1
    )

    sin = torch.repeat_interleave(
        sin,
        2,
        dim=-1
    )

    cos = cos[:x.shape[-2], :]
    sin = sin[:x.shape[-2], :]

    return (
        x * cos.unsqueeze(0).unsqueeze(0)
        +
        rotate_half(x)
        * sin.unsqueeze(0).unsqueeze(0)
    )