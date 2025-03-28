import haiku as hk
import jax
import jax.numpy as jnp
import os
from dotenv import load_dotenv

load_dotenv()

staf_params = jnp.load(os.getenv("STAF_PARAMS_PATH"))


class FINERLayer(hk.Module):
    def __init__(self, in_f, out_f, w0=200, bs_scale=5, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.ws_range = 1 / in_f if self.is_first else jnp.sqrt(6 / in_f) / w0
        self.bs_scale = bs_scale

    def gen_scale(self, x):
        return jnp.abs(x) + 1

    def __call__(self, x):
        if self.is_first:
            x = hk.Linear(
                output_size=self.out_f,
                w_init=hk.initializers.RandomUniform(-self.ws_range, self.ws_range),
                b_init=hk.initializers.RandomUniform(-self.bs_scale, self.bs_scale),
            )(x)
        else:
            x = hk.Linear(
                output_size=self.out_f,
                w_init=hk.initializers.RandomUniform(-self.ws_range, self.ws_range),
            )(x)

        return x + 0.5 if self.is_last else self.w0 * self.gen_scale(x)


class FINER(hk.Module):
    def __init__(self, w0, bs_scale, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0
        self.bs_scale = bs_scale
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = FINERLayer(
            x.shape[-1], self.width, is_first=True, w0=self.w0, bs_scale=self.bs_scale
        )(x)
        x = jnp.sin(x)

        for _ in range(self.depth - 2):
            x = FINERLayer(x.shape[-1], self.width, w0=self.hidden_w0)(x)
            x = jnp.sin(x)

        out = FINERLayer(x.shape[-1], 1, w0=self.hidden_w0, is_last=True)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class STAFLayer(hk.Module):
    def __init__(
        self, in_f, out_f, layer_idx, bias=True, w0=30, is_first=False, is_last=False
    ):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.bias = bias
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.layer_idx = layer_idx

        self.nf = 5

    def __call__(self, x):
        self.ws = hk.get_parameter(
            "omegas",
            shape=(self.nf,),
            init=lambda x1, x2: staf_params["ws"][self.layer_idx],
        )
        self.phis = hk.get_parameter(
            "phis",
            shape=(self.nf,),
            init=lambda x1, x2: staf_params["phis"][self.layer_idx],
        )
        self.bs = hk.get_parameter(
            "bs",
            shape=(self.nf,),
            init=lambda x1, x2: staf_params["bs"][self.layer_idx],
        )

        x = hk.Linear(
            output_size=self.out_f,
            with_bias=self.bias,
            w_init=hk.initializers.RandomUniform(
                -1 / jnp.sqrt(self.in_f), 1 / jnp.sqrt(self.in_f)
            ),
        )(x)

        x = self.staf_activation(x)
        return x

    def staf_activation(self, x):
        x = x[:, :, None]
        x = x.repeat(self.ws.shape[0], axis=-1)

        wsx = jnp.broadcast_to(self.ws, (*x.shape[:-1], self.ws.shape[0]))
        bsx = jnp.broadcast_to(self.bs, (*x.shape[:-1], self.bs.shape[0]))
        phisx = jnp.broadcast_to(self.phis, (*x.shape[:-1], self.phis.shape[0]))

        temp = bsx * jnp.sin((wsx * x) + phisx)
        temp2 = jnp.sum(temp, axis=-1)

        return temp2


class STAF(hk.Module):
    def __init__(self, w0, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])

        x = STAFLayer(x.shape[-1], self.width, 0, is_first=True, w0=self.w0)(x)

        for i in range(1, self.depth - 1):
            x = STAFLayer(x.shape[-1], self.width, i, w0=self.hidden_w0)(x)

        out = hk.Linear(1)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class SIRENLayer(hk.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.b = 1 / in_f if self.is_first else jnp.sqrt(6 / in_f) / w0

    def __call__(self, x):
        x = hk.Linear(
            output_size=self.out_f,
            w_init=hk.initializers.RandomUniform(-self.b, self.b),
        )(x)
        return x + 0.5 if self.is_last else self.w0 * x


class SIREN(hk.Module):
    def __init__(self, w0, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0  # to change the omega_0 of SIREN !!!!
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = SIRENLayer(x.shape[-1], self.width, is_first=True, w0=self.w0)(x)
        x = jnp.sin(x)

        for _ in range(self.depth - 2):
            x = SIRENLayer(x.shape[-1], self.width, w0=self.hidden_w0)(x)
            x = jnp.sin(x)

        out = SIRENLayer(x.shape[-1], 1, w0=self.hidden_w0, is_last=True)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class MLP(hk.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.width = width
        self.depth = depth

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = hk.Linear(self.width)(x)
        x = jax.nn.relu(x)

        for _ in range(self.depth - 2):
            x = hk.Linear(self.width)(x)
            x = jax.nn.relu(x)

        out = hk.Linear(1)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


class FFN(hk.Module):
    def __init__(self, sigma, width, depth):
        super().__init__()
        self.sigma = sigma
        self.width = width
        self.depth = depth
        self.B = self.sigma * jax.random.normal(jax.random.PRNGKey(7), (width, 2))

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = input_mapping_fourier(x, self.B)

        for _ in range(self.depth - 2):
            x = hk.Linear(self.width)(x)
            x = jax.nn.relu(x)

        out = hk.Linear(1)(x)
        out = jnp.reshape(out, list(sh[:-1]) + [1])

        return out


def input_mapping_fourier(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.0 * jnp.pi * x) @ B.T
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
