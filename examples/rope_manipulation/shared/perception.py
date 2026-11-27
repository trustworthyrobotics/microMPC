"""JAX visual encoding and Markov-state construction."""

from __future__ import annotations
import time
from collections import deque
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch

SOLVER_DTYPE = jnp.float32
PERCEPTION_DTYPE = jnp.bfloat16


def select_jax_device(device: torch.device) -> jax.Device:
    """Select the JAX device corresponding to the Torch simulation device."""
    platform = "gpu" if device.type == "cuda" else "cpu"
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        raise RuntimeError(f"JAX has no usable {platform!r} backend.") from exc
    if not devices:
        raise RuntimeError(f"JAX has no usable {platform!r} device.")
    index = device.index or 0
    if index >= len(devices):
        raise RuntimeError(
            f"Requested {device}, but JAX exposes only {len(devices)} {platform} device(s)."
        )
    return devices[index]


def _torch_numpy(tensor: torch.Tensor, dtype: np.dtype = np.float32) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def _linear_jax(x: jax.Array, params: tuple[jax.Array, jax.Array]) -> jax.Array:
    weight, bias = params
    return x @ weight.T + bias


def _layer_norm_jax(
    x: jax.Array, weight: jax.Array, bias: jax.Array, epsilon: float
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * lax.rsqrt(variance + epsilon) * weight + bias


class JAXVisionEncoder:
    """BF16 JAX implementation of the saved Hugging Face ViT and projector."""

    def __init__(
        self, model: torch.nn.Module, device: jax.Device, img_size: int
    ) -> None:
        encoder = model.encoder
        config = encoder.config
        self.device = device
        self.img_size = int(img_size)
        self.patch_size = int(config.patch_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.hidden_size) // self.num_heads
        self.layer_norm_eps = float(config.layer_norm_eps)

        def bf16(tensor: torch.Tensor) -> jax.Array:
            value = jnp.asarray(_torch_numpy(tensor), dtype=PERCEPTION_DTYPE)
            return jax.device_put(value, device)

        embeddings = encoder.embeddings
        self.cls_token = bf16(embeddings.cls_token)
        self.position_embeddings = bf16(embeddings.position_embeddings)
        projection = embeddings.patch_embeddings.projection
        # Torch convolution weights are OIHW; JAX's NHWC convolution uses HWIO.
        patch_weight = np.transpose(_torch_numpy(projection.weight), (2, 3, 1, 0))
        self.patch_weight = jax.device_put(
            jnp.asarray(patch_weight, dtype=PERCEPTION_DTYPE), device
        )
        self.patch_bias = bf16(projection.bias)

        layers = []
        for layer in encoder.encoder.layer:
            attention = layer.attention.attention
            layers.append(
                {
                    "ln1_w": bf16(layer.layernorm_before.weight),
                    "ln1_b": bf16(layer.layernorm_before.bias),
                    "query": (bf16(attention.query.weight), bf16(attention.query.bias)),
                    "key": (bf16(attention.key.weight), bf16(attention.key.bias)),
                    "value": (bf16(attention.value.weight), bf16(attention.value.bias)),
                    "attention_out": (
                        bf16(layer.attention.output.dense.weight),
                        bf16(layer.attention.output.dense.bias),
                    ),
                    "ln2_w": bf16(layer.layernorm_after.weight),
                    "ln2_b": bf16(layer.layernorm_after.bias),
                    "intermediate": (
                        bf16(layer.intermediate.dense.weight),
                        bf16(layer.intermediate.dense.bias),
                    ),
                    "output": (
                        bf16(layer.output.dense.weight),
                        bf16(layer.output.dense.bias),
                    ),
                }
            )
        self.layers = tuple(layers)
        self.final_ln = (bf16(encoder.layernorm.weight), bf16(encoder.layernorm.bias))

        projector = model.projector.net
        batch_norm = projector[1]
        self.projector_in = (bf16(projector[0].weight), bf16(projector[0].bias))
        self.projector_norm = tuple(
            bf16(value)
            for value in (
                batch_norm.weight,
                batch_norm.bias,
                batch_norm.running_mean,
                batch_norm.running_var,
            )
        )
        self.projector_eps = float(batch_norm.eps)
        self.projector_out = (bf16(projector[3].weight), bf16(projector[3].bias))
        self._preprocess_jit = jax.jit(self._preprocess_impl, device=device)
        self._encode_jit = jax.jit(self._encode_impl, device=device)

    def _preprocess_impl(self, frames: jax.Array) -> jax.Array:
        pixels = frames.astype(PERCEPTION_DTYPE) / jnp.asarray(
            255.0, dtype=PERCEPTION_DTYPE
        )
        if frames.shape[1:3] != (self.img_size, self.img_size):
            pixels = jax.image.resize(
                pixels,
                (frames.shape[0], self.img_size, self.img_size, 3),
                method="linear",
            ).astype(PERCEPTION_DTYPE)
        mean = jnp.asarray([0.485, 0.456, 0.406], dtype=PERCEPTION_DTYPE)
        std = jnp.asarray([0.229, 0.224, 0.225], dtype=PERCEPTION_DTYPE)
        return (pixels - mean) / std

    def _encode_impl(self, pixels: jax.Array) -> jax.Array:
        tokens = lax.conv_general_dilated(
            pixels,
            self.patch_weight,
            window_strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        tokens = tokens + self.patch_bias
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        cls_token = jnp.broadcast_to(
            self.cls_token, (tokens.shape[0], 1, tokens.shape[-1])
        )
        hidden = jnp.concatenate((cls_token, tokens), axis=1) + self.position_embeddings

        for layer in self.layers:
            residual = hidden
            normalized = _layer_norm_jax(
                hidden, layer["ln1_w"], layer["ln1_b"], self.layer_norm_eps
            )
            query = _linear_jax(normalized, layer["query"])
            key = _linear_jax(normalized, layer["key"])
            value = _linear_jax(normalized, layer["value"])
            shape = (hidden.shape[0], hidden.shape[1], self.num_heads, self.head_dim)
            query = query.reshape(shape).transpose(0, 2, 1, 3)
            key = key.reshape(shape).transpose(0, 2, 1, 3)
            value = value.reshape(shape).transpose(0, 2, 1, 3)
            scores = jnp.einsum("bhqd,bhkd->bhqk", query, key)
            scores = scores / jnp.asarray(self.head_dim**0.5, dtype=PERCEPTION_DTYPE)
            probabilities = jax.nn.softmax(scores, axis=-1).astype(PERCEPTION_DTYPE)
            context = jnp.einsum("bhqk,bhkd->bhqd", probabilities, value)
            context = context.transpose(0, 2, 1, 3).reshape(hidden.shape)
            hidden = residual + _linear_jax(context, layer["attention_out"])

            residual = hidden
            normalized = _layer_norm_jax(
                hidden, layer["ln2_w"], layer["ln2_b"], self.layer_norm_eps
            )
            intermediate = jax.nn.gelu(
                _linear_jax(normalized, layer["intermediate"]), approximate=False
            ).astype(PERCEPTION_DTYPE)
            hidden = residual + _linear_jax(intermediate, layer["output"])

        hidden = _layer_norm_jax(hidden, *self.final_ln, self.layer_norm_eps)
        projected = _linear_jax(hidden[:, 0], self.projector_in)
        scale, offset, running_mean, running_variance = self.projector_norm
        projected = (projected - running_mean) * lax.rsqrt(
            running_variance + self.projector_eps
        )
        projected = projected * scale + offset
        projected = jax.nn.gelu(projected, approximate=False).astype(PERCEPTION_DTYPE)
        return _linear_jax(projected, self.projector_out).astype(PERCEPTION_DTYPE)

    def encode(
        self, frames: np.ndarray, batch_size: int, *, timed: bool = False
    ) -> jax.Array | tuple[jax.Array, float]:
        preprocessed = []
        for start in range(0, len(frames), batch_size):
            pixels = jax.device_put(
                np.ascontiguousarray(frames[start : start + batch_size]), self.device
            )
            preprocessed.append(self._preprocess_jit(pixels))
        # Exclude rendering, host-to-device transfer, resize, and normalization.
        jax.block_until_ready(preprocessed)
        encode_started_ns = time.perf_counter_ns()
        embeddings = []
        for pixels in preprocessed:
            embeddings.append(self._encode_jit(pixels))
        result = jnp.concatenate(embeddings, axis=0)
        result.block_until_ready()
        elapsed_ms = (time.perf_counter_ns() - encode_started_ns) / 1e6
        return (result, elapsed_ms) if timed else result


def make_markov_state_jax(history: deque[jax.Array], markov_order: int) -> jax.Array:
    if len(history) < markov_order + 1:
        raise ValueError(f"Need {markov_order + 1} embeddings, got {len(history)}.")
    differences = jnp.stack(tuple(history)[-(markov_order + 1) :], axis=0)
    components = [differences[-1]]
    for _ in range(markov_order):
        differences = differences[1:] - differences[:-1]
        components.append(differences[-1])
    return jnp.concatenate(components, axis=-1).astype(SOLVER_DTYPE)


def jax_to_torch(array: jax.Array) -> torch.Tensor:
    """Share a JAX result with Torch through DLPack without a host round trip."""
    return torch.utils.dlpack.from_dlpack(array)
