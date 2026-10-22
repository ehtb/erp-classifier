"""
Multimodal Tabular-Text ERP Classifier.

Fuses text embeddings (vendor, description) with continuous numerical features
(z_score(net_amount), log1p(net_amount)) and categorical metadata (department)
to learn GL account classification without hardcoded business rules.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ERPClassifierModel(PreTrainedModel):
    """
    Multimodal classifier that fuses:
    - Text: Vendor + Description (via DistilBERT)
    - Continuous: [z_score(net_amount), log1p(net_amount)] -> piecewise threshold MLP
    - Categorical: department -> learned embedding
    """

    def __init__(
        self,
        config,
        model_name: Optional[str] = None,
        num_depts: Optional[int] = None,
        num_continuous: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
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
        self.encoder = AutoModel.from_pretrained(self.model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT
        self.text_norm = nn.LayerNorm(hidden_size)

        # 2. Continuous numerical projection branch
        # Crucial: Do NOT put LayerNorm before the first GELU, otherwise collinear
        # 2D projections have their magnitude normalized away before threshold activation.
        self.num_proj = nn.Sequential(
            nn.Linear(self.num_continuous, 64),
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

        # 4. Joint interaction classification head
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

        # Continuous branch: ensure float32 [B, 2] -> [B, 64]
        num_vec = self.num_proj(num_features.float())

        # Categorical branch: ensure long [B] -> [B, 32]
        dept_vec = self.dept_norm(self.dept_act(self.dept_emb(dept_id.long())))

        # Multimodal fusion: [B, 864] -> [B, num_labels]
        fused = torch.cat([cls_vec, num_vec, dept_vec], dim=-1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
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
