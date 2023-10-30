from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn


def uniform_spectral_init(r_min=0.0, r_max=1.0, max_phase=6.28):
    def init(key, shape, dtype=jnp.float32):
        key1, key2 = jax.random.split(key)
        u1 = jax.random.uniform(key1, shape, dtype)
        u2 = jax.random.uniform(key2, shape, dtype)

        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = jnp.log(max_phase * u2)

        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
        return {"nu_log": nu_log, "theta_log": theta_log, "gamma_log": gamma_log}

    return init


@jax.jit
def complex_matmul_c(C_re, C_im, x_re, x_im):
    x_re = C_re @ x_re - C_im @ x_im
    return x_re


@jax.jit
def complex_mul(A, x):
    A_re, A_im = A
    x_re, x_im = x
    res_re = A_re * x_re - A_im * x_im
    res_im = A_im * x_re + A_re * x_im
    return res_re, res_im


def tuple_to_complex(x):
    return x[0] + 1j * x[1]


def complex_to_tuple(x):
    return x.real, x.imag


def binary_operator_diag_opt(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j

    a = complex_mul(a_i, a_j)

    res = complex_mul(a_j, bu_i)
    res_re = res[0] + bu_j[0]
    res_im = res[1] + bu_j[1]

    return a, (res_re, res_im)


class LRU(nn.Module):
    d_model: int
    ssm_size: int = 64
    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    attention_dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        p = self.param(
            "diagonalised_A",
            uniform_spectral_init(
                r_min=self.r_min, r_max=self.r_max, max_phase=self.max_phase
            ),
            (self.ssm_size,),
        )

        self.nu_log, self.theta_log, self.gamma_log = (
            p["nu_log"],
            p["theta_log"],
            p["gamma_log"],
        )

        self.B_re = self.param(
            "B_re",
            nn.initializers.normal(stddev=1 / jnp.sqrt(2 * self.d_model)),
            (self.ssm_size, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            nn.initializers.normal(stddev=1 / jnp.sqrt(2 * self.d_model)),
            (self.ssm_size, self.d_model),
        )

        self.C_re = self.param(
            "C_re",
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.d_model)),
            (self.d_model, self.ssm_size),
        )
        self.C_im = self.param(
            "C_im",
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.d_model)),
            (self.d_model, self.ssm_size),
        )

        self.D = self.param("D", nn.initializers.normal(), (self.d_model,))
        self.dropout = nn.Dropout(self.attention_dropout_rate)

    def __call__(self, input_sequence, training):
        def apply_lru(Lambda_re, Lambda_im, B_re, B_im, C_re, C_im, input_sequence):

            Bu_re_elements = jax.vmap(lambda u: B_re @ u)(input_sequence)
            Bu_im_elements = jax.vmap(lambda u: B_im @ u)(input_sequence)

            elements = (
                (Lambda_re, Lambda_im),
                (Bu_re_elements, Bu_im_elements)
            )

            _, inner_states = jax.lax.associative_scan(binary_operator_diag_opt, elements)
            # inner_states.real, inner_states.imag
            return jax.vmap(partial(complex_matmul_c, C_re, C_im))(*inner_states), inner_states

        def call(input_sequence):
            deterministic = not training
            nu_log, theta_log, B_re, B_im, C_re, C_im, D, input_sequence = nn.dtypes.promote_dtype(
                self.nu_log,
                self.theta_log,
                self.B_re,
                self.B_im,
                self.C_re,
                self.C_im,
                self.D,
                input_sequence,
                dtype=self.dtype
            )
            r = jnp.exp(-jnp.exp(nu_log))

            radians = jnp.exp(theta_log)

            gamma = jnp.sqrt(1 - r ** 2)

            Lambda_re = jnp.cos(radians) * r
            Lambda_im = jnp.sin(radians) * r
            Lambda_re = jnp.repeat(Lambda_re[None, ...], repeats=input_sequence.shape[0], axis=0)
            Lambda_im = jnp.repeat(Lambda_im[None, ...], repeats=input_sequence.shape[0], axis=0)

            mask = self.dropout(
                jnp.ones_like(input_sequence[0]), deterministic=deterministic
            )
            input_sequence = jax.vmap(lambda x: mask * x)(input_sequence)

            ys, inner_states = apply_lru(
                Lambda_re,
                Lambda_im,
                B_re * jnp.expand_dims(gamma, axis=-1),
                B_im * jnp.expand_dims(gamma, axis=-1),
                C_re,
                C_im,
                input_sequence,
            )
            D, input_sequence = nn.dtypes.promote_dtype(
                self.D, input_sequence, dtype=self.dtype
            )
            Du = jax.vmap(lambda u: D * u)(input_sequence)
            res = ys + Du
            return res
        return jax.vmap(call)(input_sequence)


__all__ = ["LRU"]
