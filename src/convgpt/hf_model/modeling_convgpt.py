# coding=utf-8
# Copyright 2026 Mariusz Kurman, MedIT Solutions Sp. z o.o, Poland. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch import nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers import GenerationMixin
from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from einops import rearrange
from torch import nn, einsum
from transformers.integrations import use_kernel_forward_from_hub
from copy import deepcopy
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .configuration_convgpt import ConvGPTConfig

logger = logging.get_logger(__name__)

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(q.dtype)


@use_kernel_forward_from_hub("RMSNorm")
class ConvGPTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class ConvGPTScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, scale: float = None):
        super().__init__()
        self.scale = dim**-0.5 if scale is None else scale
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class ConvGPTScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(
                1,
            )
        )
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, n: int = None):
        if n is None:
            n = x.shape[1]
        device = x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)
        sinu = einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class ConvGPTT5RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j 1 -> i j")
        return bias * self.scale


class ConvGPTRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ConvGPTOffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.bias = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        out = einsum("... d, h d -> ... h d", x, self.weight) + self.bias
        return out.unbind(dim=-2)


class ConvGPTReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class ConvGPTLaplacianAttnFn(nn.Module):
    """https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared"""

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5


class ConvGPTAttention(nn.Module):
    def __init__(self, config: ConvGPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.transformer_dim // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.layer_type = config.layer_types[layer_idx]

        self.q_proj = nn.Linear(
            config.transformer_dim,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.transformer_dim,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.transformer_dim,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.transformer_dim,
            bias=config.attention_bias,
        )
        self.q_norm = ConvGPTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = ConvGPTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.max_window_layers = config.max_window_layers if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2).contiguous()
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2).contiguous()
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        if (
            attention_mask is not None
            and attention_mask.shape[-2] != key_states.shape[-2]
        ):
            attention_mask = nn.functional.pad(
                attention_mask,
                (0, key_states.shape[-2] - attention_mask.shape[-2]),
                value=1,
            )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ConvGPTMLP(nn.Module):
    def __init__(
        self,
        config,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = input_dim or config.transformer_dim
        self.output_size = output_dim or config.transformer_dim
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ConvGPTDecoder(nn.Module):
    def __init__(self, config: ConvGPTConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        self.self_attn = ConvGPTAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.mlp = ConvGPTMLP(config)

        self.norm1 = ConvGPTRMSNorm(config.transformer_dim)
        self.norm2 = ConvGPTRMSNorm(config.transformer_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ConvGPTPreTrainedModel(PreTrainedModel):
    config_class = ConvGPTConfig
    base_model_prefix = "convgpt"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ConvGPTDecoder"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ConvGPTModel(ConvGPTPreTrainedModel):
    def __init__(self, config: ConvGPTConfig):
        super().__init__(config)
        self.config = config
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embedding_dim_factor = config.emb_dim_factor
        self.emb_factor = config.emb_factor

        k = config.conv_kernel_h_w

        self.conv = nn.Conv2d(
            self.embedding_dim_factor,
            self.embedding_dim_factor,
            kernel_size=(1, k),
            dilation=(1, 1),
            bias=False,
        )

        self.pool = nn.AvgPool2d((self.emb_factor, self.emb_factor))
        self.norm_emb = ConvGPTScaleNorm(config.transformer_dim)

        self.conv_activation = ACT2FN[config.conv_activation]

        self.dropout = nn.Dropout(config.dropout)

        self.rotary_pos_emb = ConvGPTRotaryEmbedding(config=config)

        self.layers = nn.ModuleList(
            [
                ConvGPTDecoder(
                    config=config,
                    layer_idx=i,
                )
                for i in range(0, config.num_hidden_layers)
            ]
        )


        if config.tie_word_embeddings:
            self.scale_up = ConvGPTMLP(
                config,
                input_dim=config.transformer_dim,
                output_dim=config.hidden_size,
                intermediate_size=config.hidden_size * 4,
            )

            self.norm = ConvGPTRMSNorm(config.hidden_size)
        else:
            self.norm = ConvGPTRMSNorm(self.config.transformer_dim)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **loss_kwargs,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dtype = input_ids.dtype

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch, tokens = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch, tokens = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = 0

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embedding_dim_factor,
            self.embedding_dim_factor,
        ).transpose(1, 2).contiguous()

        pad_w = (self.conv.kernel_size[1] - 1) * self.conv.dilation[1]
        hidden_states = nn.functional.pad(hidden_states, (pad_w, 0, 0, 0))

        hidden_dtype = hidden_states.dtype
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = self.pool(hidden_states)

        hidden_states = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.shape[2]
            * hidden_states.shape[3],
        )
        hidden_states = self.norm_emb(hidden_states)

        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.layer_type],
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.layer_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        if hasattr(self, "scale_up"):
            hidden_states = self.scale_up(hidden_states)
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ConvGPTForCausalLM(ConvGPTPreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: ConvGPTConfig):
        super().__init__(config)
        self.model = ConvGPTModel(config)
        self.hidden_size = (
            config.transformer_dim
            if not config.tie_word_embeddings
            else config.hidden_size
        )
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return loss, output, past_key_values

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
