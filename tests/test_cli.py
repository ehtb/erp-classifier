import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, DistilBertConfig, DistilBertTokenizer

from classify_erp import ERPClassifier, ModelLoadError, parse_sweep_spec
from demo.app import evaluate_transaction
from scripts.evaluate import plot_boundary_sweep
from src.config import ID2LABEL, NUM_LABELS
from src.model import ERPClassifierModel, get_device, predict_probs


def _tiny_model(num_encoding: str = "ple") -> ERPClassifierModel:
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
    model = ERPClassifierModel(
        config=config,
        load_pretrained_encoder=False,
        num_encoding=num_encoding,
        numeric_bin_edges=[-1.0, 0.0, 1.0, 2.0] if num_encoding == "ple" else None,
        numeric_bin_feature_idx=1,
        num_depts=6,
        num_continuous=2,
    )
    model.config.__dict__["num_labels"] = NUM_LABELS
    return model.eval()


def _write_tiny_checkpoint(ckpt_dir: Path) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = _tiny_model("ple")
    model.save_pretrained(ckpt_dir)

    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "[VENDOR]", "[DESC]", "[PRICE_BAND]", "0", "1",
        "amazon", "services", "europe", "dell", "ultrasharp", "monitor", "credit", "note",
    ]
    vocab_tokens += [f"tok{i}" for i in range(64 - len(vocab_tokens))]
    vocab_path = ckpt_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens) + "\n")
    tokenizer = DistilBertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(ckpt_dir)

    (ckpt_dir / "scaler.json").write_text(
        json.dumps({"mean": [500.0, 5.5], "scale": [400.0, 1.2]})
    )
    depts = ["UNK", "ENG", "FACILITIES", "HR", "IT_OPS", "MARKETING"]
    (ckpt_dir / "dept_vocab.json").write_text(
        json.dumps({
            "dept2id": {d: i for i, d in enumerate(depts)},
            "id2dept": {str(i): d for i, d in enumerate(depts)},
        })
    )
    (ckpt_dir / "numeric_bins.json").write_text(
        json.dumps({"edges": [-1.0, 0.0, 1.0, 2.0], "feature_idx": 1})
    )
    return ckpt_dir


def test_parse_sweep_spec_ranges_and_non_divisible_steps():
    assert parse_sweep_spec("700:850:50") == [700.0, 750.0, 800.0, 850.0]
    assert parse_sweep_spec("750:800") == [750.0, 775.0, 800.0]
    # Non-divisible step sizes must not overshoot stop
    assert parse_sweep_spec("700:850:40") == [700.0, 740.0, 780.0, 820.0, 850.0]
    assert parse_sweep_spec("700:850:100") == [700.0, 800.0, 850.0]
    assert parse_sweep_spec("750, 800, 850.5") == [750.0, 800.0, 850.5]

    for invalid in ("850:700:25", "700:850:0", "700:850:-10", "", "   ", ",", " , , "):
        with pytest.raises(ValueError):
            parse_sweep_spec(invalid)


def test_forward_and_forward_with_diagnostics_match_exactly():
    model = _tiny_model("ple")

    input_ids = torch.tensor([[2, 10, 13, 3, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    num_features = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    dept_id = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        out_std = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_features=num_features,
            dept_id=dept_id,
        )
        out_diag = model.forward_with_diagnostics(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_features=num_features,
            dept_id=dept_id,
        )

    assert torch.equal(out_std.logits, out_diag["logits"])
    assert torch.allclose(torch.softmax(out_std.logits, dim=-1), out_diag["probs"])


def test_plot_boundary_sweep_supports_multiple_boundaries_and_custom_tolerance():
    curves = [
        {
            "boundary": 800.0,
            "amounts": np.array([700.0, 800.0, 900.0]),
            "p_hi": np.array([0.05, 0.55, 0.98]),
        },
        {
            "boundary": 800.0,
            "amounts": np.array([700.0, 800.0, 900.0]),
            "p_hi": np.array([0.02, 0.60, 0.99]),
        },
        {
            "boundary": 1500.0,
            "amounts": np.array([1200.0, 1400.0, 1500.0, 1600.0, 1800.0]),
            "p_hi": np.array([0.01, 0.10, 0.52, 0.90, 0.99]),
        },
    ]
    sweep = {
        "exactly_one_flip_fraction": 1.0,
        "non_monotone_fraction": 0.0,
        "median_crossing_rel_error": 0.006,
    }
    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / "multi_boundary.png"
        plot_boundary_sweep(
            out_path,
            primary_sweep=sweep,
            primary_curves=curves,
            step_tol_rel=0.05,
        )
        assert out_path.exists() and out_path.stat().st_size > 0


def test_hermetic_demo_evaluation_on_active_device_and_cutoff_validation(tmp_path):
    ckpt_dir = _write_tiny_checkpoint(tmp_path / "tiny-ckpt")

    for invalid_cutoff in (-0.1, 1.01):
        with pytest.raises(ValueError, match="confidence_cutoff"):
            ERPClassifier(model_dir=str(ckpt_dir), confidence_cutoff=invalid_cutoff, device="cpu")

    classifier = ERPClassifier(model_dir=str(ckpt_dir), device=get_device())

    # First and second calls must both succeed on MPS/CUDA/CPU without device mismatch
    md1, probs1, diag1, fig1 = evaluate_transaction(
        classifier, "Amazon Services Europe", "Dell UltraSharp Monitor", "IT_OPS", 750.0
    )
    plt.close(fig1)
    md2, probs2, diag2, fig2 = evaluate_transaction(
        classifier, "Amazon Services Europe", "Dell UltraSharp Monitor", "IT_OPS", 1190.0
    )
    plt.close(fig2)

    assert "Predicted account:" in md1
    assert "Predicted account:" in md2
    assert set(probs1) == set(ID2LABEL.values())
    assert pytest.approx(sum(probs1.values()), rel=1e-5) == 1.0

    # When band_edges are active, encoder_text and diag_md must include [PRICE_BAND]
    classifier.band_edges = [800.0]
    _, _, diag_below, fig3 = evaluate_transaction(
        classifier, "Amazon Services Europe", "Dell UltraSharp Monitor", "IT_OPS", 750.0
    )
    plt.close(fig3)
    _, _, diag_above, fig4 = evaluate_transaction(
        classifier, "Amazon Services Europe", "Dell UltraSharp Monitor", "IT_OPS", 850.0
    )
    plt.close(fig4)

    assert "[PRICE_BAND] 0 [VENDOR] Amazon Services Europe" in diag_below
    assert "[PRICE_BAND] 1 [VENDOR] Amazon Services Europe" in diag_above


def test_missing_artifact_raises_instead_of_exiting(tmp_path):
    ckpt_dir = _write_tiny_checkpoint(tmp_path / "missing-scaler")
    (ckpt_dir / "scaler.json").unlink()

    with pytest.raises(ModelLoadError, match="scaler.json"):
        ERPClassifier(model_dir=str(ckpt_dir), device="cpu")


def test_predict_probs_forces_eval_mode(tmp_path):
    ckpt_dir = _write_tiny_checkpoint(tmp_path / "predict-eval")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = ERPClassifierModel.from_pretrained(ckpt_dir)
    model.train()

    texts = ["[VENDOR] Amazon Services Europe [DESC] Dell UltraSharp Monitor"]
    feats = np.array([[500.0, 5.5]], dtype=np.float32)

    first = predict_probs(model, tokenizer, texts, feats, [1], "cpu", batch_size=1)
    second = predict_probs(model, tokenizer, texts, feats, [1], "cpu", batch_size=1)

    assert not model.training
    assert np.allclose(first, second)
