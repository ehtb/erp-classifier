import torch.nn as nn
from transformers import DistilBertConfig

from src.config import ID2LABEL, NUM_LABELS
from src.model import ERPClassifierModel
from train import build_optimizer_groups, no_decay_param_ids


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.num_proj = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.classifier = nn.Sequential(
            nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4)
        )


def _tiny_model(max_seq_length: int = 128) -> ERPClassifierModel:
    config = DistilBertConfig(
        vocab_size=64,
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        id2label={i: ID2LABEL[i] for i in range(NUM_LABELS)},
        label2id={ID2LABEL[i]: i for i in range(NUM_LABELS)},
    )
    config.__dict__["num_labels"] = NUM_LABELS
    return ERPClassifierModel(
        config=config,
        load_pretrained_encoder=False,
        num_encoding="ple",
        numeric_bin_edges=[-1.0, 0.0, 1.0, 2.0],
        numeric_bin_feature_idx=1,
        num_depts=6,
        num_continuous=2,
        max_length=max_seq_length,
    )


def test_no_decay_covers_custom_head_layernorms_and_biases():
    model = _DummyModel()
    no_decay = no_decay_param_ids(model)

    assert id(model.encoder.weight) not in no_decay
    assert id(model.encoder.bias) in no_decay
    assert id(model.num_proj[1].weight) in no_decay
    assert id(model.num_proj[1].bias) in no_decay
    assert id(model.classifier[1].weight) in no_decay
    assert id(model.classifier[1].bias) in no_decay
    assert id(model.classifier[0].weight) not in no_decay


def test_optimizer_groups_cover_each_parameter_once_and_use_no_decay():
    model = _DummyModel()
    groups = build_optimizer_groups(
        model, encoder_lr=3e-5, tabular_lr=5e-4, weight_decay=0.05
    )

    assert [group["weight_decay"] for group in groups] == [0.05, 0.0, 0.05, 0.0]
    grouped_ids = [id(p) for group in groups for p in group["params"]]
    all_ids = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(grouped_ids) == sorted(all_ids)
    assert len(grouped_ids) == len(set(grouped_ids))

    no_decay_group_ids = {id(p) for group in groups[1::2] for p in group["params"]}
    assert id(model.classifier[1].weight) in no_decay_group_ids
    assert id(model.classifier[0].weight) not in no_decay_group_ids


def test_max_seq_length_is_persisted_in_config_and_checkpoint(tmp_path):
    model = _tiny_model(max_seq_length=64)
    assert model.max_seq_length == 64
    assert model.config.max_seq_length == 64

    model.save_pretrained(tmp_path / "ckpt")
    reloaded = ERPClassifierModel.from_pretrained(tmp_path / "ckpt")
    assert reloaded.max_seq_length == 64
    assert reloaded.config.max_seq_length == 64
