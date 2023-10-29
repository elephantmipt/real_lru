import flax.linen as nn
import jax
import jax.numpy as jnp


def uniform_spectral_init(r_min=0.0, r_max=1.0, max_phase=6.28):
    def init(key, shape, dtype=jnp.float32):
        key1, key2 = jax.random.split(key)
        u1 = jax.random.uniform(key1, shape, dtype)
        u2 = jax.random.uniform(key2, shape, dtype)

        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
        return {"nu_log": nu_log, "theta_log": theta_log, "gamma_log": gamma_log}

    return init


def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j

    return a_j * a_i, a_j * bu_i + bu_j


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
        def apply_lru(Lambda, B_norm, C, input_sequence):
            Lambda_elements = jnp.repeat(
                Lambda[None, ...], input_sequence.shape[0], axis=0
            )
            Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
            elements = (Lambda_elements, Bu_elements)
            _, inner_states = jax.lax.associative_scan(binary_operator_diag, elements)
            return jax.vmap(lambda x: (C @ x).real)(inner_states), inner_states

        def call(input_sequence):
            deterministic = not training
            Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
            gamma = jnp.sqrt(1 - jnp.abs(Lambda) ** 2)
            B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(gamma, axis=-1)
            C = self.C_re + 1j * self.C_im
            mask = self.dropout(
                jnp.ones_like(input_sequence[0]), deterministic=deterministic
            )
            input_sequence = jax.vmap(lambda x: mask * x)(input_sequence)

            ys, inner_states = apply_lru(
                Lambda,
                B_norm,
                C,
                input_sequence,
            )
            Du = jax.vmap(lambda u: self.D * u)(input_sequence)
            return ys + Du

        return jax.vmap(call)(input_sequence)

__all__ = ["LRU"]