"""
Multimodal Tabular-Text ERP Classifier.

Fuses text embeddings (vendor, description) with continuous numerical features
(net_amount, log_amount) and categorical metadata (department) to learn
GL account classification without hardcoded business rules.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ERPClassifierModel(PreTrainedModel):
    """
    Multimodal classifier that fuses:
    - Text: Vendor + Description (via DistilBERT)
    - Continuous: net_amount, log1p(net_amount) → learns thresholds via backprop
    - Categorical: department → embedding

    This architecture allows the model to learn the 800 EUR capitalization
    threshold natively without hardcoding it in the data preparation.
    """

    def __init__(self, config, model_name: str = "distilbert-base-uncased",
                 num_depts: int = 10, num_continuous: int = 2):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        # Text encoder (DistilBERT)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT

        # Continuous numerical projection branch
        # Learns to recognize thresholds (800 EUR), ranges, and amount patterns
        self.num_proj = nn.Sequential(
            nn.Linear(num_continuous, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Categorical department embedding branch
        # Captures department-specific patterns (IT_OPS vs MARKETING)
        self.dept_emb = nn.Embedding(num_depts, 32, padding_idx=0)  # 0 = UNK
        self.dept_norm = nn.LayerNorm(32)
        self.dept_act = nn.GELU()

        # Text normalization
        self.text_norm = nn.LayerNorm(hidden_size)

        # Joint interaction classification head
        # Fuses all modalities to make final decision
        combined_dim = hidden_size + 64 + 32  # 768 + 64 + 32 = 864
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_labels)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize new layers with small random weights."""
        for module in [self.num_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize department embeddings
        nn.init.normal_(self.dept_emb.weight, mean=0.0, std=0.02)
        if self.dept_emb.padding_idx is not None:
            with torch.no_grad():
                self.dept_emb.weight[self.dept_emb.padding_idx].fill_(0)

    def forward(
        self,
        input_ids,
        attention_mask,
        num_features,
        dept_id,
        labels=None,
        return_dict=True,
    ):
        """
        Forward pass with multimodal fusion.

        Args:
            input_ids: [B, seq_len] - Tokenized text (vendor + description)
            attention_mask: [B, seq_len] - Attention mask
            num_features: [B, 2] - [net_amount, log1p(net_amount)]
            dept_id: [B] - Department integer index
            labels: [B] - Ground truth GL account labels
            return_dict: Whether to return SequenceClassifierOutput

        Returns:
            SequenceClassifierOutput with loss and logits
        """
        # Text branch: Encode vendor + description
        text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = self.text_norm(text_out.last_hidden_state[:, 0, :])  # [B, 768]

        # Continuous branch: Project numerical features
        # The network learns to recognize thresholds, not us hardcoding them
        num_vec = self.num_proj(num_features)  # [B, 64]

        # Categorical branch: Embed department
        dept_vec = self.dept_emb(dept_id)      # [B, 32]
        dept_vec = self.dept_norm(dept_vec)
        dept_vec = self.dept_act(dept_vec)

        # Fusion: Concatenate all modalities
        fused = torch.cat([cls_vec, num_vec, dept_vec], dim=-1)  # [B, 864]

        # Classification
        logits = self.classifier(fused)  # [B, num_labels]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + text_out[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
