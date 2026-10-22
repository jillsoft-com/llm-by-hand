"""Reproduce the five-sentence forward chain in chapters 5–7.

The initial parameters are the eight-decimal values fixed in chapter 2. All
subsequent calculations use float64, including the two Post-LN operations.
"""

import numpy as np


TOKENS = ["哈尔滨", "站", "在", "东", "南", "西", "北", "里", "边", "。", "<Pad>", "<EOS>"]
IDS = np.array([
    [0, 3, 1, 2, 3, 8, 9, 11, 10, 10],
    [0, 5, 1, 2, 5, 8, 9, 11, 10, 10],
    [0, 4, 1, 2, 4, 8, 9, 11, 10, 10],
    [0, 6, 1, 2, 6, 8, 9, 11, 10, 10],
    [0, 1, 2, 7, 8, 9, 11, 10, 10, 10],
], dtype=np.int64)
VALID = np.array([7, 7, 7, 7, 6], dtype=np.int64)
TARGETS = np.roll(IDS, -1, axis=1)
POSITION = np.column_stack((np.sin(np.arange(10)), np.cos(np.arange(10)), np.sin(np.arange(10) / 100)))


def initial_parameters():
    np.random.seed(42)  # Chapter 2 uses the legacy global generator.

    def xavier(rows, columns):
        return np.round(np.random.uniform(-np.sqrt(6 / (rows + columns)),
                                         np.sqrt(6 / (rows + columns)), (rows, columns)), 8)

    def kaiming(rows, columns):
        return np.round(np.random.uniform(-np.sqrt(6 / rows),
                                         np.sqrt(6 / rows), (rows, columns)), 8)

    wq, wk, wv = xavier(3, 3), xavier(3, 3), xavier(3, 3)
    wo, w1, w2 = xavier(3, 12), kaiming(3, 6), xavier(6, 3)
    we = np.array([
        [0.5, 0.5, 0.1], [0.6, 0.4, 0.1], [0.3, 0.7, 0.1],
        [0.9, 0.1, 0], [0.1, -0.9, 0], [-0.9, 0.1, 0],
        [0.1, 0.9, 0], [0.4, 0.6, 0.1], [0.5, -0.5, 0],
        [0.1, 0.1, 0], [0, 0, 0], [1, 1, 0.1],
    ], dtype=np.float64)
    return dict(W_E=we, W_Q=wq, W_K=wk, W_V=wv, W_1=w1,
                b_1=np.zeros(6), W_2=w2, b_2=np.zeros(3), W_O=wo,
                gamma_1=np.ones(3), beta_1=np.zeros(3),
                gamma_2=np.ones(3), beta_2=np.zeros(3))


def layer_norm(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    inv_std = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inv_std
    return normalized * gamma + beta, (normalized, inv_std)


def forward(params, ids=IDS, valid=VALID, return_cache=False):
    ids = np.asarray(ids, dtype=np.int64)
    valid = np.asarray(valid, dtype=np.int64)
    x = params["W_E"][ids] + POSITION
    q, k, v = (x @ params[name] for name in ("W_Q", "W_K", "W_V"))
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(3)
    row = np.arange(10)[None, :, None]
    col = np.arange(10)[None, None, :]
    allowed = (col <= row) & (col < valid[:, None, None])
    masked = np.where(allowed, scores, -np.inf)
    exp = np.exp(masked - masked.max(axis=-1, keepdims=True))
    a = exp / exp.sum(axis=-1, keepdims=True)
    z = a @ v
    r1 = x + z
    ln1, norm1 = layer_norm(r1, params["gamma_1"], params["beta_1"])
    pre = ln1 @ params["W_1"] + params["b_1"]
    h = np.maximum(pre, 0)
    f = h @ params["W_2"] + params["b_2"]
    r2 = ln1 + f
    ln2, norm2 = layer_norm(r2, params["gamma_2"], params["beta_2"])
    logits = ln2 @ params["W_O"]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    if return_cache:
        return probs, dict(ids=ids, x=x, q=q, k=k, v=v, a=a, z=z,
                           ln1=ln1, norm1=norm1, pre=pre, h=h,
                           ln2=ln2, norm2=norm2, logits=logits)
    return probs


def sentence_losses(probs, targets=TARGETS, valid=VALID):
    rows = np.arange(10)[None, :]
    selected = probs[np.arange(len(probs))[:, None], rows, targets]
    valid_rows = rows < valid[:, None]
    return (-np.log(selected) * valid_rows).sum(axis=1) / valid


def verify():
    params = initial_parameters()
    assert sum(x.size for x in params.values()) == 156
    probs, cache = forward(params, return_cache=True)
    losses = sentence_losses(probs)
    expected = np.array([2.894136671, 3.044410954, 2.815754664,
                         2.898048908, 2.895936438])
    np.testing.assert_allclose(losses, expected, atol=5e-9, rtol=0)
    np.testing.assert_allclose(probs.sum(axis=-1), 1, atol=2e-15, rtol=0)
    np.testing.assert_allclose(cache["a"].sum(axis=-1), 1, atol=2e-15, rtol=0)
    assert np.all(cache["a"][:, np.triu_indices(10, 1)[0], np.triu_indices(10, 1)[1]] == 0)
    print("句损失:", " ".join(f"{x:.9f}" for x in losses))
    print(f"Full Batch 损失: {losses.mean():.9f}")
    return params, losses


if __name__ == "__main__":
    verify()
