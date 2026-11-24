"""Strict symmetry (S_N, translation+PBC, O(3) rotation+reflection) and C^2 autograd tests."""

import math
import numpy as np
import pytest
import torch

from geoprobe_deepdft import CanonicalPaiNNDensityModel, GeoProbeDensityModel
from geoprobe_deepdft.graph_utils import grid_coords_from_indices, make_probe_edges_single


def _build_dummy_crystal(device, seed=0):
    rng = np.random.default_rng(seed)
    cell = torch.tensor([[5.6, 0.2, 0.0], [1.4, 4.9, 0.0], [0.0, 0.1, 13.5]], dtype=torch.float32, device=device)
    frac_atoms = torch.tensor([
        [0.10, 0.20, 0.15],
        [0.45, 0.55, 0.40],
        [0.75, 0.30, 0.65],
        [0.25, 0.80, 0.85],
    ], dtype=torch.float32, device=device)
    nums = torch.tensor([3, 28, 8, 25], dtype=torch.long, device=device)
    atom_xyz = frac_atoms @ cell

    # Build atom edges under PBC
    r_b = torch.arange(-1, 2, device=device)
    shifts = torch.cartesian_prod(r_b, r_b, r_b).to(torch.float32)
    img_pos = (atom_xyz[:, None, :] + (shifts @ cell)[None, :, :]).reshape(-1, 3)
    diff = atom_xyz[:, None, :] - img_pos[None, :, :]
    d2 = (diff * diff).sum(dim=-1)
    dst, ns = torch.nonzero((d2 > 1e-6) & (d2 < 16.0), as_tuple=True)
    src = torch.div(ns, shifts.shape[0], rounding_mode="floor")
    sidx = ns % shifts.shape[0]
    ae = torch.stack([src, dst], dim=1)
    aed = shifts[sidx]

    # Sample arbitrary real-space probes
    frac_probes = torch.tensor(rng.uniform(0.05, 0.95, size=(64, 3)), dtype=torch.float32, device=device)
    probe_xyz = frac_probes @ cell
    pe, ped = make_probe_edges_single(atom_xyz, cell, probe_xyz, cutoff=4.0, base_shifts=shifts, pbc=True)
    return nums, atom_xyz, cell, ae, aed, probe_xyz, pe, ped, shifts


def _to_batch(nums, atom_xyz, cell, ae, aed, probe_xyz, pe, ped):
    return {
        "nodes": nums.unsqueeze(0),
        "atom_xyz": atom_xyz.unsqueeze(0),
        "cell": cell.unsqueeze(0),
        "atom_edges": ae.unsqueeze(0),
        "atom_edges_displacement": aed.unsqueeze(0),
        "probe_xyz": probe_xyz.unsqueeze(0),
        "probe_edges": pe.unsqueeze(0),
        "probe_edges_displacement": ped.unsqueeze(0),
        "num_nodes": torch.tensor([len(nums)], device=nums.device),
        "num_atom_edges": torch.tensor([len(ae)], device=nums.device),
        "num_probes": torch.tensor([len(probe_xyz)], device=nums.device),
        "num_probe_edges": torch.tensor([len(pe)], device=nums.device),
    }


def _make_o3_matrix(improper=False, device="cpu"):
    rng = np.random.default_rng(17 if not improper else 99)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    if improper:
        q[:, 0] *= -1.0
    return torch.tensor(q, dtype=torch.float32, device=device)


def test_parameter_counts():
    can = CanonicalPaiNNDensityModel(3, 128, 4.0)
    geo = GeoProbeDensityModel(3, 128, 4.0)
    assert sum(p.numel() for p in can.parameters() if p.requires_grad) == 1487233
    assert sum(p.numel() for p in geo.parameters() if p.requires_grad) == 1871617


def test_exact_symmetries_loaded_checkpoint():
    device = torch.device("cpu")
    model = GeoProbeDensityModel(3, 128, 4.0).to(device).eval()
    sd = torch.load("checkpoints/geoprobe_h128_seed42.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd["model"], strict=True)

    nums, atom_xyz, cell, ae, aed, probe_xyz, pe, ped, shifts = _build_dummy_crystal(device)
    with torch.no_grad():
        ref = model(_to_batch(nums, atom_xyz, cell, ae, aed, probe_xyz, pe, ped)).squeeze(0)

    # 1. Atom permutation (S_N)
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long, device=device)
    inv_perm = torch.argsort(perm)
    ae_perm = torch.stack([inv_perm[ae[:, 0]], inv_perm[ae[:, 1]]], dim=1)
    pe_perm = torch.stack([inv_perm[pe[:, 0]], pe[:, 1]], dim=1)
    with torch.no_grad():
        out_perm = model(_to_batch(nums[perm], atom_xyz[perm], cell, ae_perm, aed, probe_xyz, pe_perm, ped)).squeeze(0)
    assert torch.max(torch.abs(out_perm - ref)).item() < 1e-5

    # 2. Uniform Cartesian translation
    shift_vec = torch.tensor([3.1, -2.4, 5.7], dtype=torch.float32, device=device)
    with torch.no_grad():
        out_tr = model(_to_batch(nums, atom_xyz + shift_vec, cell, ae, aed, probe_xyz + shift_vec, pe, ped)).squeeze(0)
    assert torch.max(torch.abs(out_tr - ref)).item() < 1e-5

    # 3. Periodic lattice wrap of atom 0 by +cell[0] - cell[1]
    lat_shift = cell[0] - cell[1]
    atom_xyz_wrap = atom_xyz.clone()
    atom_xyz_wrap[0] = atom_xyz_wrap[0] + lat_shift
    # Recompute periodic edges equivalently
    r_b = torch.arange(-2, 3, device=device)
    sh5 = torch.cartesian_prod(r_b, r_b, r_b).to(torch.float32)
    img_pos = (atom_xyz_wrap[:, None, :] + (sh5 @ cell)[None, :, :]).reshape(-1, 3)
    diff = atom_xyz_wrap[:, None, :] - img_pos[None, :, :]
    d2 = (diff * diff).sum(dim=-1)
    dst_w, ns_w = torch.nonzero((d2 > 1e-6) & (d2 < 16.0), as_tuple=True)
    ae_w = torch.stack([torch.div(ns_w, sh5.shape[0], rounding_mode="floor"), dst_w], dim=1)
    aed_w = sh5[ns_w % sh5.shape[0]]
    pe_w, ped_w = make_probe_edges_single(atom_xyz_wrap, cell, probe_xyz, 4.0, sh5, pbc=True)
    with torch.no_grad():
        out_pbc = model(_to_batch(nums, atom_xyz_wrap, cell, ae_w, aed_w, probe_xyz, pe_w, ped_w)).squeeze(0)
    assert torch.max(torch.abs(out_pbc - ref)).item() < 1e-5

    # 4. Proper rotation SO(3) and improper reflection O(3) \ SO(3)
    for improper in (False, True):
        R = _make_o3_matrix(improper=improper, device=device)
        with torch.no_grad():
            out_rot = model(_to_batch(nums, atom_xyz @ R.T, cell @ R.T, ae, aed, probe_xyz @ R.T, pe, ped)).squeeze(0)
        assert torch.max(torch.abs(out_rot - ref)).item() < 1e-5


def test_c2_smoothness_and_autograd():
    device = torch.device("cpu")
    model = GeoProbeDensityModel(3, 128, 4.0).to(device).eval()
    sd = torch.load("checkpoints/geoprobe_h128_seed42.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd["model"], strict=True)

    nums, atom_xyz, cell, ae, aed, probe_xyz, pe, ped, _ = _build_dummy_crystal(device)
    px = probe_xyz.clone().detach().requires_grad_(True)
    batch = _to_batch(nums, atom_xyz, cell, ae, aed, px, pe, ped)
    pred = model(batch)
    grad1 = torch.autograd.grad(pred.sum(), px, create_graph=True)[0]
    assert torch.isfinite(grad1).all()
    assert grad1.abs().max().item() > 1e-6

    grad2_loss = (grad1 * grad1).sum()
    grad2 = torch.autograd.grad(grad2_loss, px)[0]
    assert torch.isfinite(grad2).all()
