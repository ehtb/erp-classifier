"""
Multimodal Tabular-Text ERP Classifier.

Fuses text embeddings (vendor, description) with continuous numerical features
(z_score(net_amount), log1p(net_amount)) and categorical metadata (department)
to learn GL account classification without hardcoded business rules.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class PeriodicNumericalEncoding(nn.Module):
    """
    Learnable periodic encoding for numerical features.

    Provides high-frequency resolution around boundaries (e.g., 800 EUR threshold)
    without hardcoding. Concatenates [x, sin(2π*x*freqs), cos(2π*x*freqs)].
    """

    def __init__(self, input_dim: int = 2, n_bins: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.n_bins = n_bins

        # Learnable frequency parameters (initialized with log-spaced values)
        init_freqs = torch.logspace(-1, 2, n_bins)  # 0.1 to 100
        self.freqs = nn.Parameter(init_freqs.unsqueeze(0).repeat(input_dim, 1))

        # Output: input_dim * (1 + 2*n_bins) = 2 * (1 + 32) = 66
        self.out_dim = input_dim * (1 + 2 * n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            [B, input_dim * (1 + 2*n_bins)]
        """
        # x: [B, 2]
        batch_size = x.shape[0]

        # Expand: [B, 2, 1] * [2, 16] -> [B, 2, 16]
        x_expanded = x.unsqueeze(-1) * self.freqs.unsqueeze(0)

        # Periodic features
        sin_features = torch.sin(2 * math.pi * x_expanded)  # [B, 2, 16]
        cos_features = torch.cos(2 * math.pi * x_expanded)  # [B, 2, 16]

        # Concatenate raw + periodic per dimension: [B, 2, 1+2*16] -> [B, 2, 33]
        features = torch.cat([x.unsqueeze(-1), sin_features, cos_features], dim=-1)

        # Flatten: [B, 2*33] = [B, 66]
        return features.view(batch_size, -1)


class ERPClassifierModel(PreTrainedModel):
    """
    Multimodal classifier that fuses:
    - Text: Vendor + Description (via DistilBERT)
    - Continuous: [z_score(net_amount), log1p(net_amount)] -> piecewise threshold MLP
    - Categorical: department -> learned embedding
    """

    config_class = AutoConfig

    @property
    def all_tied_weights_keys(self):
        """Return tied weights keys (required by newer transformers)."""
        return {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # When loading from checkpoint, don't re-download the encoder
        kwargs['load_pretrained_encoder'] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(
        self,
        config,
        model_name: Optional[str] = None,
        num_depts: Optional[int] = None,
        num_continuous: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        load_pretrained_encoder: bool = True,
    ):
        super().__init__(config)

        # Persist hyperparameters in config so save_pretrained / from_pretrained works
        self.model_name = model_name or getattr(config, "base_model_name", "distilbert-base-uncased")
        self.num_depts = num_depts or getattr(config, "num_depts", 10)
        self.num_continuous = num_continuous or getattr(config, "num_continuous", 2)
        self.num_labels = config.num_labels

        config.base_model_name = self.model_name
        config.num_depts = self.num_depts
        config.num_continuous = self.num_continuous

        # 1. Text encoder branch
        encoder_config = AutoConfig.from_pretrained(self.model_name)
        if load_pretrained_encoder:
            # Training from scratch - load pretrained weights
            self.encoder = AutoModel.from_pretrained(self.model_name)
        else:
            # Loading from checkpoint - use from_config, state_dict will load weights
            self.encoder = AutoModel.from_config(encoder_config)

        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT
        self.text_norm = nn.LayerNorm(hidden_size)

        # 2. Continuous numerical projection branch with periodic encoding
        # Periodic encoding provides high-frequency resolution around 800 EUR boundary
        self.periodic_encoding = PeriodicNumericalEncoding(input_dim=self.num_continuous, n_bins=16)
        periodic_dim = self.periodic_encoding.out_dim  # 2 * (1 + 2*16) = 66

        self.num_proj = nn.Sequential(
            nn.Linear(periodic_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        # 3. Categorical department embedding branch (index 0 = UNK, trainable)
        self.dept_emb = nn.Embedding(self.num_depts, 32)
        self.dept_act = nn.GELU()
        self.dept_norm = nn.LayerNorm(32)

        # 4. FiLM gating: tabular/categorical features modulate text embeddings
        # Provides multiplicative interaction: text * (1 + gamma) + beta
        tabular_dim = 64 + 32  # num_vec + dept_vec
        self.film_gate = nn.Linear(tabular_dim, hidden_size * 2)  # -> (gamma, beta)

        # 5. Modality dropout probability (for UNK dept_id training)
        self.modality_dropout_p = 0.15

        # 6. Joint interaction classification head
        combined_dim = hidden_size + 64 + 32  # 864
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_labels),
        )

        # Register class_weights as a buffer so it moves to GPU/MPS automatically
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

        self._init_custom_heads()

    def _init_custom_heads(self):
        """Initialize non-pretrained tabular and classifier layers."""
        for module in [self.num_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.normal_(self.dept_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_features: torch.Tensor,
        dept_id: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # Text branch: [B, 768]
        text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = self.text_norm(text_out.last_hidden_state[:, 0, :])

        # Continuous branch with periodic encoding: [B, 2] -> [B, 66] -> [B, 64]
        periodic_features = self.periodic_encoding(num_features.float())
        num_vec = self.num_proj(periodic_features)

        # Categorical branch with modality dropout (15% -> UNK during training)
        dept_id_input = dept_id.long()
        if self.training:
            # Randomly mask dept_id to 0 (UNK) with 15% probability
            mask = torch.rand(dept_id_input.shape[0], device=dept_id_input.device) < self.modality_dropout_p
            dept_id_input = dept_id_input.masked_fill(mask, 0)

        dept_vec = self.dept_norm(self.dept_act(self.dept_emb(dept_id_input)))

        # FiLM gating: tabular features modulate text embeddings
        # gamma, beta = split(tabular -> 2*hidden_size)
        tabular_vec = torch.cat([num_vec, dept_vec], dim=-1)  # [B, 96]
        film_params = self.film_gate(tabular_vec)  # [B, 1536]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # [B, 768] each

        # Modulate: cls_vec * (1 + tanh(gamma)) + beta
        cls_vec = cls_vec * (1.0 + torch.tanh(gamma)) + beta

        # Multimodal fusion: [B, 864] -> [B, num_labels]
        fused = torch.cat([cls_vec, num_vec, dept_vec], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            # Label smoothing 0.05 to prevent overconfidence
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.05)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        if not return_dict:
            output = (logits,) + text_out[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
