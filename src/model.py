"""
Multimodal Tabular-Text ERP Classifier.

Fuses text embeddings (vendor, description) with continuous numerical features
(z_score(net_amount), log1p(net_amount)) and categorical metadata (department)
to learn GL account classification without hardcoded business rules.
"""

from pathlib import Path
from typing import Optional
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import NUM_LABELS


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


class PiecewiseLinearEncoding(nn.Module):
    """
    Piecewise-linear encoding (PLE) of one numeric feature.

    Each bin b contributes clip((x - e_b) / (e_{b+1} - e_b), 0, 1): a ramp that is 0
    below the bin, 1 above it and linear inside. The ramps are monotone in x and
    fixed (no learnable frequencies), so the head can express a step at any bin edge
    with a handful of weights. Edges are learned from training data by the dataset
    builder (quantiles or target-aware tree splits), never hard-coded here.
    The raw features are appended so the head still sees the continuous value.
    """

    def __init__(self, edges, feature_idx: int, input_dim: int):
        super().__init__()
        if edges is None or len(edges) < 2:
            raise ValueError("PiecewiseLinearEncoding requires at least 2 bin edges")
        edges = torch.tensor(edges, dtype=torch.float32)
        self.register_buffer("lower", edges[:-1])
        self.register_buffer("width", (edges[1:] - edges[:-1]).clamp_min(1e-6))
        self.feature_idx = feature_idx
        self.out_dim = input_dim + len(edges) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x[:, self.feature_idx].unsqueeze(-1)
        ramps = ((v - self.lower) / self.width).clamp(0.0, 1.0)
        return torch.cat([x, ramps], dim=-1)


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
        use_class_weights: Optional[bool] = None,
        load_pretrained_encoder: bool = True,
        num_encoding: Optional[str] = None,
        numeric_bin_edges: Optional[list] = None,
        numeric_bin_feature_idx: Optional[int] = None,
    ):
        super().__init__(config)

        # Persist hyperparameters in config so save_pretrained / from_pretrained works
        self.model_name = model_name or getattr(config, "base_model_name", "distilbert-base-uncased")
        self.num_depts = num_depts or getattr(config, "num_depts", 10)
        self.num_continuous = num_continuous or getattr(config, "num_continuous", 2)
        # `num_labels` is a property derived from id2label in transformers 5.x, and
        # older checkpoints were saved with a literal null. Fall back to id2label
        # rather than crashing on nn.Linear(..., None).
        self.num_labels = getattr(config, "num_labels", None) or len(
            getattr(config, "id2label", None) or {}
        ) or NUM_LABELS
        if use_class_weights is None:
            use_class_weights = getattr(config, "use_class_weights", True)
        self.use_class_weights = bool(use_class_weights)
        config.use_class_weights = self.use_class_weights

        self.num_encoding = num_encoding or getattr(config, "num_encoding", "periodic")
        self.numeric_bin_edges = numeric_bin_edges or getattr(config, "numeric_bin_edges", None)
        idx = numeric_bin_feature_idx if numeric_bin_feature_idx is not None else getattr(config, "numeric_bin_feature_idx", 1)

        config.num_encoding = self.num_encoding
        config.numeric_bin_edges = self.numeric_bin_edges
        config.numeric_bin_feature_idx = idx
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

        # 2. Continuous numerical projection branch.
        # PLE (default) gives monotone ramps with data-learned edges; the periodic
        # encoder remains available for the ablation.
        if self.num_encoding == "ple":
            self.periodic_encoding = PiecewiseLinearEncoding(
                self.numeric_bin_edges, feature_idx=idx, input_dim=self.num_continuous
            )
        else:
            self.periodic_encoding = PeriodicNumericalEncoding(input_dim=self.num_continuous, n_bins=16)
        periodic_dim = self.periodic_encoding.out_dim  # periodic: 2 * (1 + 2*16) = 66

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

        # Always register a buffer (ones when unused) so checkpoints saved with and
        # without class weights load without an UNEXPECTED-key warning.
        if class_weights is None:
            class_weights = torch.ones(self.num_labels)
        self.register_buffer("class_weights", class_weights.float())

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
        sample_weight: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # Text branch: [B, 768]
        text_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = self.text_norm(text_out.last_hidden_state[:, 0, :])

        # Continuous branch: [B, 2] -> [B, 18 or 66] -> [B, 64]
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
            # Label smoothing 0.05 to prevent overconfidence. Class weights are used
            # only when the checkpoint was trained with them; counterfactual-aware
            # training instead uses per-sample weights that restore the source prior.
            weight = self.class_weights if self.use_class_weights else None
            if sample_weight is not None:
                per_sample = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1).long(),
                    weight=weight,
                    label_smoothing=0.05,
                    reduction="none",
                )
                w = sample_weight.view(-1).float()
                loss = (per_sample * w).sum() / w.sum().clamp_min(1e-6)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1).long(),
                    weight=weight,
                    label_smoothing=0.05,
                )

        if not return_dict:
            output = (logits,) + text_out[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def get_device() -> torch.device:
    """Select CUDA, MPS, or CPU device for inference/evaluation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: Path, device: torch.device):
    """Load an ERPClassifierModel, its tokenizer, and scaler statistics from `model_dir`."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ERPClassifierModel.from_pretrained(model_dir)
    model.to(device).eval()

    with open(model_dir / "scaler.json") as f:
        scaler = json.load(f)
    return model, tokenizer, np.array(scaler["mean"]), np.array(scaler["scale"])


@torch.no_grad()
def predict_probs(model, tokenizer, texts, num_features, dept_ids, device, batch_size: int = 64) -> np.ndarray:
    """Softmax probabilities for parallel lists of inputs."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_features=torch.tensor(np.asarray(num_features[i:i + batch_size]), dtype=torch.float32, device=device),
            dept_id=torch.tensor(dept_ids[i:i + batch_size], dtype=torch.long, device=device),
        ).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out)
