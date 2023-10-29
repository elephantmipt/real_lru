import time

import jax
from jax import numpy as jnp

from flax.training.train_state import TrainState

import optax

from real_lru.real import LRU
from real_lru.naive import LRU as LRU_naive



### TEST if outputs match ###

key = jax.random.PRNGKey(907654)
batch = jax.random.normal(key, (1, 1024, 256), dtype=jnp.float32)

lru_opt = LRU(
    d_model=256
)
lru = LRU_naive(
    d_model=256
)

params = lru_opt.init(key, batch, False)
res_opt = lru_opt.apply(params, batch, False)
res = lru.apply(params, batch, False)

assert jnp.allclose(res, res_opt, atol=1e-6)

def create_train_state(cls_model, batch_size, hidden_dim, ssm_dim, seq_len, dtype, key):

    model = cls_model(
        d_model=hidden_dim,
        ssm_size=ssm_dim,
        dtype=dtype
    )
    batch = jax.random.normal(key, (batch_size, seq_len, hidden_dim), dtype=dtype)
    params = model.init(key, batch, training=True)

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(params)

    train_state = TrainState(
        apply_fn=model.apply,
        params=params,
        step=0,
        tx=optimizer,
        opt_state=opt_state
    )
    return train_state

import numpy as np
from tqdm import trange

def get_benchmark(train_state, batch_size, seq_len, hidden_dim, dtype):

    @jax.jit
    def forward(batch, train_state):
        out = train_state.apply_fn(train_state.params, batch, True)
        return out

    def test_speed(train_state, bs, seq_len, hid_state, key, dtype):
        batch = jax.random.normal(key, (bs, seq_len, hid_state), dtype=dtype)
        start = time.time()
        out = forward(batch, train_state)
        jax.block_until_ready(out)
        assert out.dtype == dtype
        return start

    test_times = []
    key = jax.random.PRNGKey(907654)
    for i in trange(201):
        key, _ = jax.random.split(key)
        start = time.time()
        test_speed(train_state, batch_size, seq_len, hidden_dim, key, dtype)
        delta = time.time() - start
        if i > 1:
            test_times += [delta]
    print(f"\n {np.mean(test_times):.4f} ± {np.std(test_times):.4f}")
    return test_times


if __name__ == "__main__":
    results = []
    model_cls = [LRU, LRU_naive]
    hidden_dim = 256
    ssm_dim = 128
    batch_size = 64

    for model in model_cls:
        for seq_len in [512, 1024, 2048, 4096]:
            ts = create_train_state(model, batch_size, hidden_dim, ssm_dim, seq_len, jnp.float32, jax.random.PRNGKey(42))
            results.append(get_benchmark(ts, batch_size, seq_len, hidden_dim, jnp.float32))

    for seq_len in [512, 1024, 2048, 4096]:
        ts = create_train_state(LRU, batch_size, hidden_dim, ssm_dim, seq_len, jnp.float16, jax.random.PRNGKey(42))
        results.append(get_benchmark(ts, batch_size, seq_len, hidden_dim, jnp.float16))

    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set(style="whitegrid", font_scale=1.2)
    x = [512, 1024, 2048, 4096]
    plt.plot(x, np.mean(results[4:8], -1), label="Naive LRU")
    plt.plot(x, np.mean(results[:4], -1), label="LRU Real")
    plt.plot(x, np.mean(results[-4:], -1), label="LRU Real + FP16")
    plt.legend()
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (seconds)")
    plt.show()
