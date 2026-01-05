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

from transformers.utils import logging
import math
from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation

from typing import Optional

logger = logging.get_logger(__name__)

CONVGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


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

        self.emb_dim_factor = int(math.sqrt(self.hidden_size))

        if self.emb_dim_factor**2 != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be a perfect square. Got {math.sqrt(self.hidden_size)} as sqrt."
            )

        self.emb_factor = int(math.sqrt(self.emb_dim_factor)) // 2

        self.transformer_dim = int(
            self.hidden_size / (self.emb_factor * self.emb_factor)
        )

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]

        layer_type_validation(self.layer_types)

        super().__init__(
            **kwargs,
        )
