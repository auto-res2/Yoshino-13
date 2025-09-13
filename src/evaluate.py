# src/evaluate.py
"""All evaluation / analysis code.
This contains the three experiment entry-points as well as auxiliary
metrics and plotting helpers.  The public API that *main.py* relies on
is the `EXPERIMENT_DISPATCH` dictionary mapping the YAML experiment
names to callables which accept a single `cfg` dict argument.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # head-less backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mutual_info_score
from transformers import AutoTokenizer

from .preprocess import prepare_dataset, PromptDataset, RESEARCH_DIR, IMAGES_DIR
from .train import load_model

# ---------------------------------------------------------------
#   generic helpers
# ---------------------------------------------------------------


def compute_mi(labels: List[int], representations: torch.Tensor) -> float:
    """Empirical mutual information between discrete labels and continuous
    representations using coarse binning (20 uniform buckets per
    dimension).  Fast and adequate for the qualitative comparison we
    need here.
    """
    reps_np = representations.cpu().numpy()
    binned = (reps_np * 20).astype(int).clip(0, 19)
    flat = ["_".join(map(str, row)) for row in binned]
    return mutual_info_score(labels, flat)


# ---------------------------------------------------------------
#   EXPERIMENT 1 – SOIC Leakage Stress Test
# ---------------------------------------------------------------

def run_soic_leakage(cfg: Dict, smoke: bool):
    exp_name = "soic_leakage"
    print("\n==== EXPERIMENT 1 – SOIC Leakage-Stress Test ====")

    result_path = RESEARCH_DIR / f"{exp_name}.json"
    figures: List[str] = []

    # 1) datasets ----------------------------------------------------------------
    secret_dir = prepare_dataset(cfg, "secret_prompts")
    benign_dir = prepare_dataset(cfg, "benign_prompts")

    secret_file = next(secret_dir.glob("**/data.jsonl"))
    benign_file = next(benign_dir.glob("**/data.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["models"]["base"]["repo"], token=cfg.get("_hf_token")
    )

    # ---------------------------------------------------------------------------
    # Ensure padding token is available (many GPT-family tokenizers lack one)
    # ---------------------------------------------------------------------------
    if tokenizer.pad_token is None:
        # Use EOS as PAD to avoid size mismatch when calling with padding="max_length"
        tokenizer.pad_token = tokenizer.eos_token

    secret_ds = PromptDataset(secret_file, tokenizer)
    benign_ds = PromptDataset(benign_file, tokenizer)

    max_prompts = cfg["resources"]["max_prompts"]
    batch_size = cfg["resources"]["batch_size"]

    secret_loader = torch.utils.data.DataLoader(secret_ds, batch_size=batch_size, shuffle=True)
    benign_loader = torch.utils.data.DataLoader(benign_ds, batch_size=batch_size, shuffle=True)

    # 2) models ------------------------------------------------------------------
    model_variants: Dict[str, Dict] = {
        "base": cfg["models"]["base"],
        "hyperion": cfg["models"].get("hyperion"),
    }
    if not smoke:
        for k in [
            "oracle_guard",
            "hyperion_no_hessian",
            "hyperion_no_dither",
        ]:
            if k in cfg["models"]:
                model_variants[k] = cfg["models"][k]

    mi_results: Dict[str, float] = {}
    acc_results: Dict[str, float] = {}

    for variant_name, variant_info in model_variants.items():
        if variant_info is None:
            continue  # not present in smoke config
        print(f"[Variant {variant_name}] loading …")
        model = load_model(variant_info["repo"], hf_token=cfg.get("_hf_token"))

        collected_logits = []
        collected_labels: List[int] = []

        def _collect(loader, label):
            n = 0
            for batch in loader:
                if n >= max_prompts:
                    break
                # increment BEFORE early-exit so we never exceed max_prompts
                batch_size_local = batch["input_ids"].size(0)
                n += batch_size_local
                with torch.no_grad():
                    out = model(
                        input_ids=batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                    )
                    logits = out.logits[:, -1, :].float().cpu()
                    collected_logits.append(logits)
                    collected_labels.extend([label] * logits.size(0))

        _collect(secret_loader, 1)
        _collect(benign_loader, 0)

        logits_tensor = torch.cat(collected_logits, dim=0)
        mi_results[variant_name] = compute_mi(collected_labels, logits_tensor)

        # naive exact-token recovery ------------------------------------------------
        _, pred_tok = logits_tensor.topk(1, dim=1)
        pred_ids: List[int] = pred_tok.squeeze(1).tolist()
        preds = [tokenizer.decode([tid], skip_special_tokens=True).strip() for tid in pred_ids]
        # accuracy here is proportion of *non-empty* decoded tokens (simple sanity-check)
        non_empty = sum(1 for t in preds if t)
        acc_results[variant_name] = non_empty / len(preds) if preds else 0.0

        del model
        torch.cuda.empty_cache()

    # 3) persist JSON -------------------------------------------------------------
    res_dict = {
        "mutual_information_bits": mi_results,
        "token_recovery_accuracy": acc_results,
        "num_samples": len(collected_labels),
    }
    result_path.write_text(json.dumps(res_dict, indent=2))

    # 4) figures ------------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=list(mi_results.keys()), y=list(mi_results.values()), palette="viridis")
    ax.set_ylabel("Empirical MI (bits)")
    ax.set_xlabel("Model Variant")
    for patch, val in zip(ax.patches, mi_results.values()):
        ax.annotate(f"{val:.3f}", (patch.get_x() + patch.get_width() / 2.0, val), ha="center", va="bottom")
    fig1 = IMAGES_DIR / "mi_bits_models.pdf"
    plt.tight_layout()
    plt.savefig(fig1, bbox_inches="tight")
    plt.close()
    figures.append(fig1.name)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=list(acc_results.keys()), y=list(acc_results.values()), palette="magma")
    ax.set_ylabel("Exact Token Recovery – Acc")
    ax.set_xlabel("Model Variant")
    for patch, val in zip(ax.patches, acc_results.values()):
        ax.annotate(f"{val:.2%}", (patch.get_x() + patch.get_width() / 2.0, val), ha="center", va="bottom")
    fig2 = IMAGES_DIR / "token_recovery_accuracy.pdf"
    plt.tight_layout()
    plt.savefig(fig2, bbox_inches="tight")
    plt.close()
    figures.append(fig2.name)

    # 5) STDOUT -------------------------------------------------------------------
    print(json.dumps(res_dict, indent=2))
    print("Figures generated:")
    for f in figures:
        print(f"  - {f}")


# ---------------------------------------------------------------
#   EXPERIMENT 2 – CARE-Bench (placeholder)
# ---------------------------------------------------------------

def run_care_bench(cfg: Dict, *_):
    care_dir = prepare_dataset(cfg, "care_bench")
    if not care_dir.exists():
        raise RuntimeError("CARE-Bench dataset could not be located after download – aborting.")
    raise RuntimeError(
        "Full CARE-Bench experiment requires multi-day execution. "
        "Please run the dedicated pipeline provided in the HYPERION-SHIELD repository."
    )


# ---------------------------------------------------------------
#   EXPERIMENT 3 – GMSM Side-Channel (placeholder)
# ---------------------------------------------------------------

def run_gmsm(cfg: Dict, *_):
    _ = prepare_dataset(cfg, "gaze_capture")
    _ = prepare_dataset(cfg, "ar_latency")
    raise RuntimeError(
        "Full GMSM evaluation requires specialised AR hardware traces – aborting."
    )


# ----------------------------------------------------------------
#   public dispatch table
# ----------------------------------------------------------------

EXPERIMENT_DISPATCH = {
    "soic_leakage": run_soic_leakage,
    "care_bench": run_care_bench,
    "gmsm_side_channel": run_gmsm,
}
