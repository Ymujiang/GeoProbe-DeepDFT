"""Hardware throughput and peak memory profiler for GeoProbe-DeepDFT vs Canonical PaiNN."""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geoprobe_deepdft import CanonicalPaiNNDensityModel, GeoProbeDensityModel
from geoprobe_deepdft.graph_utils import grid_coords_from_indices, make_probe_edges_single


def main():
    p = argparse.ArgumentParser(description="Profile inference latency, throughput, and peak memory")
    p.add_argument("--data", default="sample_data/nmc_sample_2structs.pt")
    p.add_argument("--n_probes", type=int, default=4096)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    rec = torch.load(args.data, map_location="cpu", weights_only=False)["records"][0]
    pos = torch.as_tensor(rec["positions"], dtype=torch.float32, device=device)
    cell = torch.as_tensor(rec["cell"], dtype=torch.float32, device=device)
    nums = torch.as_tensor(rec["numbers"], dtype=torch.long, device=device)
    ae = torch.as_tensor(rec["atom_edges"], dtype=torch.long, device=device)
    aed = torch.as_tensor(rec["atom_edges_displacement"], dtype=torch.float32, device=device)

    base_r = torch.arange(-1, 2, device=device)
    base_shifts = torch.cartesian_prod(base_r, base_r, base_r).to(torch.float32)
    idx = torch.arange(args.n_probes, device=device)
    p_xyz = grid_coords_from_indices(rec["grid_shape"], cell, idx)
    pe, ped = make_probe_edges_single(pos, cell, p_xyz, 4.0, base_shifts, pbc=True)

    batch = {
        "nodes": nums.unsqueeze(0), "atom_xyz": pos.unsqueeze(0), "cell": cell.unsqueeze(0),
        "atom_edges": ae.unsqueeze(0), "atom_edges_displacement": aed.unsqueeze(0),
        "num_nodes": torch.tensor([len(nums)], device=device), "num_atom_edges": torch.tensor([len(ae)], device=device),
        "probe_xyz": p_xyz.unsqueeze(0), "probe_edges": pe.unsqueeze(0), "probe_edges_displacement": ped.unsqueeze(0),
        "num_probes": torch.tensor([args.n_probes], device=device), "num_probe_edges": torch.tensor([len(pe)], device=device),
    }

    configs = [
        ("canonical_h128", CanonicalPaiNNDensityModel, 128),
        ("param_matched_h144", CanonicalPaiNNDensityModel, 144),
        ("geoprobe_h128", GeoProbeDensityModel, 128),
    ]
    out = {"device": str(device), "n_probes": args.n_probes, "n_edges": int(len(pe)), "models": {}}
    for name, Cls, h in configs:
        net = Cls(3, h, 4.0).to(device).eval()
        with torch.no_grad():
            sa, va = net.atom_model(batch)
            for _ in range(5):
                _ = net.probe_model(batch, sa, va)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(args.iters):
                _ = net.probe_model(batch, sa, va)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000.0 / args.iters
            vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
        out["models"][name] = {
            "n_params": sum(x.numel() for x in net.parameters() if x.requires_grad),
            "ms_per_chunk": round(ms, 3),
            "probes_per_s": round(args.n_probes / (ms / 1000.0), 1),
            "peak_vram_mb": round(vram_mb, 2),
        }
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
