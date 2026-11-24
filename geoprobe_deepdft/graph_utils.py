"""Periodic boundary condition (PBC) neighbor-search helpers matching asap3 edge parity."""

import torch


def make_probe_edges_single(atom_xyz, cell, probe_xyz, cutoff=4.0, base_shifts=None, pbc=True):
    """Build directed atom->probe edges within ``cutoff`` under general triclinic PBC."""
    device = atom_xyz.device
    dtype = atom_xyz.dtype
    if not pbc:
        shifts = torch.zeros((1, 3), device=device, dtype=dtype)
    else:
        vol = torch.abs(torch.det(cell))
        c01 = torch.linalg.norm(torch.cross(cell[1], cell[2], dim=0))
        c12 = torch.linalg.norm(torch.cross(cell[2], cell[0], dim=0))
        c20 = torch.linalg.norm(torch.cross(cell[0], cell[1], dim=0))
        heights = torch.stack([vol / c01, vol / c12, vol / c20])
        if base_shifts is not None and torch.all(heights >= cutoff):
            shifts = base_shifts.to(device=device, dtype=dtype)
        else:
            max_s = torch.ceil(cutoff / torch.clamp(heights, min=1e-4)).to(torch.int64) + 1
            r0 = torch.arange(-max_s[0], max_s[0] + 1, device=device)
            r1 = torch.arange(-max_s[1], max_s[1] + 1, device=device)
            r2 = torch.arange(-max_s[2], max_s[2] + 1, device=device)
            shifts = torch.cartesian_prod(r0, r1, r2).to(dtype)

    n_shifts = shifts.shape[0]
    img_pos = (atom_xyz[:, None, :] + (shifts @ cell)[None, :, :]).reshape(-1, 3)
    diff = probe_xyz[:, None, :] - img_pos[None, :, :]
    dist_sq = (diff * diff).sum(dim=-1)
    p_idx, ns_idx = torch.nonzero(dist_sq < (cutoff * cutoff), as_tuple=True)
    a_idx = torch.div(ns_idx, n_shifts, rounding_mode="floor")
    s_idx = ns_idx % n_shifts
    return torch.stack([a_idx, p_idx], dim=1), shifts[s_idx]


def grid_coords_from_indices(grid_shape, cell, flat_idx):
    """Map flattened VASP-style CHGCAR grid indices to Cartesian coordinates."""
    ny, nz = int(grid_shape[1]), int(grid_shape[2])
    nx = int(grid_shape[0])
    ix = torch.div(flat_idx, ny * nz, rounding_mode="floor")
    rem = flat_idx % (ny * nz)
    iy = torch.div(rem, nz, rounding_mode="floor")
    iz = rem % nz
    frac = torch.stack([ix / nx, iy / ny, iz / nz], dim=1).to(cell.dtype)
    return frac @ cell
