"""Evaluate GeoProbe-DeepDFT or Canonical PaiNN checkpoints on real-space density grids."""

import argparse
import json
import os
import time
import numpy as np
import torch

from geoprobe_deepdft import CanonicalPaiNNDensityModel, GeoProbeDensityModel
from geoprobe_deepdft.graph_utils import grid_coords_from_indices, make_probe_edges_single


def evaluate_structures(model, records, device, cutoff=4.0, chunk=4096):
    model.eval()
    base_r = torch.arange(-1, 2, device=device)
    base_shifts = torch.cartesian_prod(base_r, base_r, base_r).to(torch.float32)

    eps_maes, maes, rmses = [], [], []
    per_struct = []
    t0 = time.time()

    with torch.inference_mode():
        for rec in records:
            pos = torch.as_tensor(rec["positions"], dtype=torch.float32, device=device)
            cell = torch.as_tensor(rec["cell"], dtype=torch.float32, device=device)
            nums = torch.as_tensor(rec["numbers"], dtype=torch.long, device=device)
            ae = torch.as_tensor(rec["atom_edges"], dtype=torch.long, device=device)
            aed = torch.as_tensor(rec["atom_edges_displacement"], dtype=torch.float32, device=device)
            dens = torch.as_tensor(rec["density"], dtype=torch.float32, device=device)
            gshape = rec["grid_shape"]
            pbc = bool(np.all(rec["pbc"]))

            batch = {
                "nodes": nums.unsqueeze(0),
                "atom_xyz": pos.unsqueeze(0),
                "cell": cell.unsqueeze(0),
                "atom_edges": ae.unsqueeze(0),
                "atom_edges_displacement": aed.unsqueeze(0),
                "num_nodes": torch.tensor([len(nums)], device=device),
                "num_atom_edges": torch.tensor([len(ae)], device=device),
            }
            s_a, v_a = model.atom_model(batch)
            npts = int(dens.numel())
            all_idx = torch.arange(npts, device=device)
            all_xyz = grid_coords_from_indices(gshape, cell, all_idx)

            abs_err = 0.0
            sq_err = 0.0
            target_sum = 0.0
            for st in range(0, npts, chunk):
                p_sub = all_xyz[st:st + chunk]
                pe, ped = make_probe_edges_single(pos, cell, p_sub, cutoff, base_shifts, pbc=pbc)
                batch["probe_xyz"] = p_sub.unsqueeze(0)
                batch["probe_edges"] = pe.unsqueeze(0)
                batch["probe_edges_displacement"] = ped.unsqueeze(0)
                batch["num_probes"] = torch.tensor([len(p_sub)], device=device)
                batch["num_probe_edges"] = torch.tensor([len(pe)], device=device)
                pred = model.probe_model(batch, s_a, v_a).squeeze(0)
                diff = dens[st:st + chunk] - pred
                abs_err += torch.abs(diff).sum().item()
                sq_err += (diff * diff).sum().item()
                target_sum += dens[st:st + chunk].sum().item()

            eps = 100.0 * abs_err / max(target_sum, 1e-12)
            mae = abs_err / npts
            rmse = float(np.sqrt(sq_err / npts))
            eps_maes.append(eps)
            maes.append(mae)
            rmses.append(rmse)
            per_struct.append({
                "filename": rec.get("filename", f"struct_{len(per_struct)}"),
                "eps_mae_pct": round(eps, 6),
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
            })

    return {
        "n_structures": len(records),
        "mean_eps_mae_pct": round(float(np.mean(eps_maes)), 4),
        "mae": round(float(np.mean(maes)), 6),
        "rmse": round(float(np.mean(rmses)), 6),
        "elapsed_s": round(time.time() - t0, 2),
        "per_struct": per_struct,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate GeoProbe-DeepDFT electron density model")
    p.add_argument("--model", choices=["geoprobe", "canonical"], default="geoprobe")
    p.add_argument("--node_size", type=int, default=128)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", default="sample_data/nmc_sample_2structs.pt")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    ckpt_path = args.ckpt or (
        "checkpoints/geoprobe_h128_seed42.pt" if args.model == "geoprobe" else "checkpoints/canonical_h128_seed42.pt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    Cls = GeoProbeDensityModel if args.model == "geoprobe" else CanonicalPaiNNDensityModel
    model = Cls(3, args.node_size, 4.0).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    res = evaluate_structures(model, records, device, cutoff=4.0, chunk=args.chunk)
    res["model"] = args.model
    res["node_size"] = args.node_size
    res["n_params"] = n_params
    res["device"] = str(device)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
