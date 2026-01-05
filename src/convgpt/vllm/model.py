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

from typing import List, Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

import math
from dataclasses import dataclass

@dataclass
class ConvGPTConfig(PretrainedConfig):
    model_type = "convgpt"
    vocab_size: int = 32768
    hidden_size: int = 1296
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 128
    num_hidden_layers: int = 24
    laplace_attn_fn: bool = True
    conv_activation: str = "silu"
    conv_filters: int = 4
    number_of_conv: int = 1
    conv_kernel_h_w: int = 3
    scale: float = 2.0
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    attention_activation: str = "silu"
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_cache: bool = True
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    is_causal: bool = True
    max_seq_length: Optional[int] = None
    num_experts: int = 20
    num_of_flash_experts: int = 4
    expert_gate_r: int = 4
    top_k_experts: int = 4
    use_moe_lm_head: bool = False
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    layer_types: Optional[list] = None
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = None

    def __init__(self, **kwargs):
        hs = kwargs.get("hidden_size", self.hidden_size)
        
        # Calculate defaults based on hs
        default_emb_dim = int(math.sqrt(hs))
        default_emb_fact = int(math.sqrt(default_emb_dim)) // 2

        self.emb_dim_factor = kwargs.get("emb_dim_factor", default_emb_dim)
        self.emb_factor = kwargs.get("emb_factor", default_emb_fact)

        self.transformer_dim = int(hs / (self.emb_factor * self.emb_factor)) if self.emb_factor > 0 else hs
        super().__init__(**kwargs)


class ConvGPTSCaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, scale: float = None):
        super().__init__()
        self.scale = dim**-0.5 if scale is None else scale
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class ConvGPTMLP(nn.Module):
    def __init__(
        self,
        config: ConvGPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.transformer_dim
        self.intermediate_size = config.intermediate_size
        
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class ConvGPTAttention(nn.Module):
    def __init__(
        self,
        config: ConvGPTConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.transformer_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        
        # Support TP
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        self.qkv_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[
                self.total_num_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim,
                self.total_num_kv_heads * self.head_dim
            ],
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position = getattr(config, "max_position_embeddings", 8192)
        rope_scaling = getattr(config, "rope_scaling", None)
        
        rope_params = {"rope_theta": rope_theta, "rope_type": "default"}
        if rope_scaling is not None:
             rope_params["rope_scaling"] = rope_scaling

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            is_neox_style=True,
            rope_parameters=rope_params,
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)
        
        q, k = self.rotary_emb(positions, q, k)
        
        attn_output = self.attn(q, k, v)
        
        output, _ = self.o_proj(attn_output)
        return output


class ConvGPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ConvGPTConfig,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = ConvGPTAttention(
            config, 
            layer_idx, 
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = ConvGPTMLP(
            config, 
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.norm1 = RMSNorm(config.transformer_dim, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.transformer_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states, residual = self.norm1(hidden_states, residual)
            
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        
        # Post-attention norm
        hidden_states, residual = self.norm2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class ConvGPTModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens" if prefix else "embed_tokens",
        )
        
        k = config.conv_kernel_h_w
        self.conv = nn.Conv2d(
            config.emb_dim_factor,
            config.emb_dim_factor,
            kernel_size=(1, k),
            dilation=(1, 1),
            bias=False,
        )
        self.pool = nn.AvgPool2d((config.emb_factor, config.emb_factor))
        self.norm_emb = ConvGPTSCaleNorm(config.transformer_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            ConvGPTDecoderLayer(
                config, 
                i, 
                vllm_config,
                prefix=f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        ])
        
        if config.tie_word_embeddings:
            self.scale_up_gate_up = MergedColumnParallelLinear(
                 input_size=config.transformer_dim,
                 output_sizes=[config.hidden_size * 4] * 2, 
                 bias=False,
                 quant_config=quant_config,
                 prefix=f"{prefix}.scale_up_gate_up" if prefix else "scale_up_gate_up",
            )
            self.scale_up_down = RowParallelLinear(
                 input_size=config.hidden_size * 4,
                 output_size=config.hidden_size,
                 bias=False,
                 quant_config=quant_config,
                 prefix=f"{prefix}.scale_up_down" if prefix else "scale_up_down",
            )
            self.act_fn = SiluAndMul()
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = RMSNorm(config.transformer_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Conv & Pool Logic
        num_tokens, hidden_size = hidden_states.shape
        emb_dim = self.config.emb_dim_factor

        
        hidden_states = hidden_states.view(num_tokens, emb_dim, emb_dim)
        hidden_states = hidden_states.permute(1, 0, 2).unsqueeze(0)
        
        pad_w = (self.conv.kernel_size[1] - 1) * self.conv.dilation[1]
        hidden_states = nn.functional.pad(hidden_states, (pad_w, 0, 0, 0))
        
        hidden_states = self.conv(hidden_states)
        
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = self.pool(hidden_states)
        
        hidden_states = hidden_states.squeeze(0).flatten(1)
        
        hidden_states = self.norm_emb(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual
            )
            
        if hasattr(self, "scale_up_gate_up"):             
             if residual is not None:
                 hidden_states = hidden_states + residual
             
             gate_up, _ = self.scale_up_gate_up(hidden_states)
             x = self.act_fn(gate_up)
             hidden_states, _ = self.scale_up_down(x)
             hidden_states = self.norm(hidden_states)
        else:
             hidden_states, _ = self.norm(hidden_states, residual)
             
        return hidden_states


class ConvGPTForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = ConvGPTModel(vllm_config, prefix=model_prefix)
        
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size if config.tie_word_embeddings else config.transformer_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", 0),
            (".qkv_proj", ".k_proj", 1),
            (".qkv_proj", ".v_proj", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            ("model.scale_up_gate_up", "model.scale_up.gate_proj", 0),
            ("model.scale_up_gate_up", "model.scale_up.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name: continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                     if "scale_up" in weight_name:
                         mapped_name = name.replace("model.scale_up.gate_proj", "model.scale_up_gate_up")
                         mapped_name = mapped_name.replace("model.scale_up.up_proj", "model.scale_up_gate_up")
                     else:
                         mapped_name = name.replace(weight_name, param_name)
                     
                     if mapped_name not in params_dict:
                         continue
                     
                     param = params_dict[mapped_name]
                     weight_loader = getattr(param, "weight_loader", default_weight_loader)
                     weight_loader(param, loaded_weight, shard_id)
                     break
            else:
                if "model.scale_up.down_proj" in name:
                     name = name.replace("model.scale_up.down_proj", "model.scale_up_down")
                
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

