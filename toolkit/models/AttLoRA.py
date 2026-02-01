# Copyright 2023-present the HuggingFace Inc. team.
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


# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling LoRWeB or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions in the LICENSE file located at the root directory.

from typing import List, Optional, Dict, Type, Union, Tuple
import math
import torch
import weakref
from optimum.quanto import QBytesTensor
import torch.nn as nn
from toolkit.network_mixins import ToolkitModuleMixin

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
]
CONV_MODULES = [
]

Network = Union['LoRASpecialNetwork']
Module = Union['LoRAModule', 'AttLoRAModule']
ExtractMode = Union[
    'existing'
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]

class AttLoRAModule(ToolkitModuleMixin, torch.nn.Module):
    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = False  # True for DoRA
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        # ToolkitModuleMixin.__init__(self, network=network)
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self._multiplier: Union[float, list, torch.Tensor] = None

        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.orig_module_ref = weakref.ref(org_module)
        self.scalar = torch.tensor(1.0, device=org_module.weight.device)
        # check if parent has bias. if not force use_bias to False
        if org_module.bias is None:
            use_bias = False

        if org_module.__class__.__name__ in CONV_MODULES:
            raise NotImplementedError("Convolutional layers are not supported yet")

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_dim = lora_dim

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        # self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=use_bias)

        self.loras_num = kwargs["loras_num"]
        self.lora_keys_dim = kwargs["lora_keys_dim"]
        self.lora_heads = kwargs["lora_heads"]
        self.lora_softmax = kwargs["lora_softmax"]
        self.mixing_coeffs_type = kwargs["mixing_coeffs_type"]
        self.query_dim = kwargs.get("query_dim", None)

        self.lora_down = nn.Parameter(torch.randn((self.loras_num, in_dim, self.lora_dim)))
        self.lora_up = nn.Parameter(torch.randn((self.loras_num, self.lora_dim, out_dim)))
        self.lora_up_bias = None
        if use_bias:
            self.lora_up_bias = nn.Parameter(torch.randn((self.loras_num, out_dim)))
        self.lora_keys = nn.Parameter(torch.randn((self.loras_num, self.lora_keys_dim)))

        self.heads_out_mat = None
        # How the inner attn happens:
        # Choice 1 - Does the inner attn query uses the input, or do we have a separate query in the forward
        # Choice 2 - Should the query be projected, and if so, how?
        self.external_query = kwargs["external_query"]
        self.query_pooling = None
        self.query_mode = kwargs["query_mode"]
        self.query_projection_type = kwargs["query_projection_type"]
        if not self.external_query:
            self.query_pooling = kwargs["pooling_type"]
            self.query_dim = in_dim
        elif self.external_query and 'cat-' in self.query_mode:
            self.query_dim *= 3

        self.att_lora_to_q = None
        if self.query_projection_type == "none":
            self.att_lora_to_q = nn.Identity()
        elif self.query_projection_type == "linear":
            self.att_lora_to_q = nn.Linear(self.query_dim, self.lora_keys_dim, bias=False)
        else:
            raise ValueError(f"Invalid query projection type: {self.query_projection_type}")

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up)
        if use_bias:
            torch.nn.init.zeros_(self.lora_up_bias)

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module

    def extract_weight(
            self: Module,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        raise NotImplementedError("AttLoRA extract_weight not implemented")
