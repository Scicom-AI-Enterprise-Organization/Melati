# Copyright 2026 Scicom (MSC) Berhad. Licensed under the Apache License, Version 2.0.
# Built on modeling_dasheng.py and configuration_dasheng.py (Copyright 2023-2024 Xiaomi Corporation and HuggingFace
# Inc. team, Apache-2.0), shipped beside this file unchanged.
"""Melati's AudioSet tagger on Dasheng: Dasheng's encoder with a head for AudioSet's 527 classes on its tokens averaged
over time. Input: DashengFeatureExtractor's mel spectrogram, [batch, 64, frames]. Output: one logit a class; a sigmoid
gives each class's probability."""
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_dasheng import DashengConfig
from .modeling_dasheng import DashengModel, DashengPreTrainedModel


class DashengAudioSetTagger(DashengPreTrainedModel):
    config_class = DashengConfig

    def __init__(self, config: DashengConfig):
        super().__init__(config)
        self.dasheng = DashengModel(config)
        d = config.encoder_kwargs["embed_dim"]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config.num_labels)
        self.post_init()

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SequenceClassifierOutput:
        h = self.dasheng(input_values).hidden_states                  # [batch, tokens, dim], 25 tokens a second
        logits = self.head(self.norm(h.mean(1)))
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=h)
