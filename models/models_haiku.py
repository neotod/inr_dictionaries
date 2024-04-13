import haiku as hk
import jax
import jax.numpy as jnp

params_path = '/run/media/neo/joint/compsci/projects/ParAcNet/parac_parameters.npz'
parac_params = jnp.load(params_path)
print(parac_params)


class PARACLayer(hk.Module):
    def __init__(self, in_f, out_f, layer_idx, bias=True, omega_0=200, is_first=False):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.bias = bias
        self.is_first = is_first
        self.omega_0 = omega_0
        self.layer_idx = layer_idx

        self.nf = 5

    def __call__(self, x):
        self.ws = hk.get_parameter(
            "weights", shape=(self.nf,), init=parac_params[self.layer_idx]["ws"]
        )
        self.phis = hk.get_parameter(
            "phis", shape=(self.nf,), init=parac_params[self.layer_idx]["phis"]
        )
        self.bs = hk.get_parameter(
            "biases", shape=(self.nf,), init=parac_params[self.layer_idx]["bs"]
        )

        x = hk.Linear(
            output_size=self.out_f,
            bias=self.bias,
            w_init=hk.initializers.RandomUniform(
                -1 / jnp.sqrt(self.in_f), 1 / jnp.sqrt(self.in_f)
            ),
        )(x)

        x = self.params_dict(x)
        return x

    def parac_activation(self, x):
        x = x[:, :, :, None].broadcast_to(self.ws.shape)
        wsx = hk.broadcast(self.ws, x.shape)
        bsx = hk.broadcast(self.bs, x.shape)
        phisx = hk.broadcast(self.phis, x.shape)
        temp = bsx * (hk.sin((wsx * x) + phisx))
        temp2 = hk.sum(temp, axis=3)
        return temp2


class PARAC(hk.Module):
    def __init__(self, w0, width, hidden_w0, depth):
        super().__init__()
        self.w0 = w0  # to change the omega_0 of PARAC !!!!
        self.width = width
        self.depth = depth
        self.hidden_w0 = hidden_w0

    def __call__(self, coords):
        sh = coords.shape
        x = jnp.reshape(coords, [-1, 2])
        x = PARACLayer(x.shape[-1], self.width, 0, is_first=True, w0=self.w0)(x)

        for i in range(self.depth - 2):
            x = PARACLayer(x.shape[-1], self.width, i + 1, w0=self.hidden_w0)(x)

        out = PARACLayer(
            x.shape[-1], 1, self.depth - 1, w0=self.hidden_w0, is_last=True
        )(x)
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
