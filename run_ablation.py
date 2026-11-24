"""Inspect parameter counts and reported benchmark results across architectural ablations."""

import argparse
import json
import os
import torch

from geoprobe_deepdft import CanonicalPaiNNDensityModel, GeoProbeDensityModel


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def module_breakdown(model):
    return {
        "atom_embeddings": sum(p.numel() for p in model.atom_model.atom_embeddings.parameters()),
        "atom_interactions": sum(p.numel() for p in model.atom_model.interactions.parameters()),
        "atom_scalar_vector_update": sum(p.numel() for p in model.atom_model.scalar_vector_update.parameters()),
        "probe_message_layers": sum(p.numel() for p in model.probe_model.message_layers.parameters()),
        "probe_scalar_vector_update": sum(p.numel() for p in model.probe_model.scalar_vector_update.parameters()),
        "probe_readout": sum(p.numel() for p in model.probe_model.readout_function.parameters()),
        "total": count_params(model),
    }


def main():
    p = argparse.ArgumentParser(description="Parameter and ablation ledger for GeoProbe-DeepDFT")
    p.add_argument("--list_params", action="store_true", help="Print parameter breakdown for all core architectures")
    args = p.parse_args()

    models = {
        "canonical_h128": CanonicalPaiNNDensityModel(3, 128, 4.0),
        "param_matched_h144": CanonicalPaiNNDensityModel(3, 144, 4.0),
        "geoprobe_h128": GeoProbeDensityModel(3, 128, 4.0),
    }
    out = {k: module_breakdown(m) for k, m in models.items()}

    if os.path.exists("results/exp2_cv5_and_ablations.json"):
        e2 = json.load(open("results/exp2_cv5_and_ablations.json"))
        out["recorded_ablations_nmc50"] = {
            k: {
                "n_params": v["n_params"],
                "best_val_mae": round(v["best_val_mae"], 6),
                "test50_eps_mae_pct": round(v["test"]["mean_eps_mae"], 4),
            }
            for k, v in e2["ablations"].items()
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
