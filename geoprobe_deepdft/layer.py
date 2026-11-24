from typing import Tuple, List
import itertools
import math
import torch
from torch import nn
import numpy as np


def pad_and_stack(tensors: List[torch.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


def shifted_softplus(x):
    """
    Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return nn.functional.softplus(x) - np.log(2.0)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return shifted_softplus(x)


def unpad_and_cat(stacked_seq: torch.Tensor, seq_len: torch.Tensor):
    """
    Unpad and concatenate by removing batch dimension

    Args:
        stacked_seq: (batch_size, max_length, *) Tensor
        seq_len: (batch_size) Tensor with length of each sequence

    Returns:
        (prod(seq_len), *) Tensor

    """
    unstacked = stacked_seq.unbind(0)
    unpadded = [
        torch.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len.unbind(0))
    ]
    return torch.cat(unpadded, dim=0)


def sum_splits(values: torch.Tensor, splits: torch.Tensor):
    """
    Sum across dimension 0 of the tensor `values` in chunks
    defined in `splits`

    Args:
        values: Tensor of shape (`prod(splits)`, *)
        splits: 1-dimensional tensor with size of each chunk

    Returns:
        Tensor of shape (`splits.shape[0]`, *)

    """
    # prepare an index vector for summation
    ind = torch.zeros(splits.sum(), dtype=splits.dtype, device=splits.device)
    ind[torch.cumsum(splits, dim=0)[:-1]] = 1
    ind = torch.cumsum(ind, dim=0)
    # prepare the output
    sum_y = torch.zeros(
        splits.shape + values.shape[1:], dtype=values.dtype, device=values.device
    )
    # do the actual summation
    sum_y.index_add_(0, ind, values)
    return sum_y


def calc_distance(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate distance of edges

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
        return_diff: If non-zero return the also the vector corresponding to edges
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = torch.sqrt(
        torch.sum(torch.square(diff), dim=1, keepdim=True)
    )  # num_edges, 1

    if return_diff:
        return dist, diff
    else:
        return dist


def calc_distance_to_probe(
    positions: torch.Tensor,
    positions_probe: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate distance of edges

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        positions_probe: Tensor of shape (num_probes, 3) with xyz coordinates of probes inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # num_edges, 3
    diff = this_pos - neigh_abs_pos  # num_edges, 3
    dist = torch.sqrt(
        torch.sum(torch.square(diff), dim=1, keepdim=True)
    )  # num_edges, 1
    if return_diff:
        return dist, diff
    else:
        return dist


def gaussian_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a number of Gaussian basis function.
    Expand_params is a list of length input_x.shape[1]

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (start, step, stop) tuples

    Returns:
        (num_edges, ``ceil((stop - start)/step)``) tensor

    """
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            start, step, stop = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            sigma = step
            basis_mu = torch.arange(
                start, stop, step, device=input_x.device, dtype=input_x.dtype
            )
            expanded_list.append(
                torch.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma ** 2))
            )
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


class SchnetMessageFunction(nn.Module):
    def __init__(self, input_size, edge_size, output_size, hard_cutoff):
        super().__init__()
        self.msg_function_edge = nn.Sequential(
            nn.Linear(edge_size, output_size),
            ShiftedSoftplus(),
            nn.Linear(output_size, output_size),
        )
        self.msg_function_node = nn.Sequential(
            nn.Linear(input_size, input_size),
            ShiftedSoftplus(),
            nn.Linear(input_size, output_size),
        )

        self.soft_cutoff_func = lambda x: 1.0 - torch.sigmoid(
            5 * (x - (hard_cutoff - 1.5))
        )

    def forward(self, node_state, edge_state, edge_distance):
        gates = self.msg_function_edge(edge_state) * self.soft_cutoff_func(
            edge_distance
        )
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class Interaction(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver=False):
        super().__init__()

        self.message_sum_module = MessageSum(
            node_size, edge_size, cutoff, include_receiver
        )

        self.state_transition_function = nn.Sequential(
            nn.Linear(node_size, node_size),
            ShiftedSoftplus(),
            nn.Linear(node_size, node_size),
        )

    def forward(self, node_state, edges, edge_state, edges_distance):

        # Compute sum of messages
        message_sum = self.message_sum_module(
            node_state, edges, edge_state, edges_distance
        )

        # State transition
        new_state = node_state + self.state_transition_function(message_sum)

        return new_state


class MessageSum(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, include_receiver):
        super().__init__()

        self.include_receiver = include_receiver

        if include_receiver:
            input_size = node_size * 2
        else:
            input_size = node_size

        self.message_function = SchnetMessageFunction(
            input_size, edge_size, node_size, cutoff
        )

    def forward(
        self, node_state, edges, edge_state, edges_distance, receiver_nodes=None
    ):
        """

        Args:
            node_state: [num_nodes, n_node_features] State of input nodes
            edges: [num_edges, 2] array of sender and receiver indices
            edge_state: [num_edges, n_features] array of edge features
            edges_distance: [num_edges, 1] array of distances
            receiver_nodes: If given, use these nodes as receiver nodes instead of node_state

        Returns:
            sum of messages to each node

        """
        # Compute all messages
        if self.include_receiver:
            if receiver_nodes is not None:
                senders = node_state[edges[:, 0]]
                receivers = receiver_nodes[edges[:, 1]]
                nodes = torch.cat((senders, receivers), dim=1)
            else:
                num_edges = edges.shape[0]
                nodes = torch.reshape(node_state[edges], (num_edges, -1))
        else:
            nodes = node_state[edges[:, 0]]  # Only include sender in messages
        messages = self.message_function(nodes, edge_state, edges_distance)

        # Sum messages
        if receiver_nodes is not None:
            message_sum = torch.zeros_like(receiver_nodes)
        else:
            message_sum = torch.zeros_like(node_state)
        message_sum.index_add_(0, edges[:, 1], messages)

        return message_sum


class EdgeUpdate(nn.Module):
    def __init__(self, edge_size, node_size):
        super().__init__()

        self.node_size = node_size
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * node_size + edge_size, 2 * edge_size),
            ShiftedSoftplus(),
            nn.Linear(2 * edge_size, edge_size),
        )

    def forward(self, edge_state, edges, node_state):
        combined = torch.cat(
            (node_state[edges].view(-1, 2 * self.node_size), edge_state), axis=1
        )
        return self.edge_update_mlp(combined)


class PaiNNUpdate(nn.Module):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""

    def __init__(self, node_size):
        super().__init__()

        self.linearU = nn.Linear(node_size, node_size, bias=False)
        self.linearV = nn.Linear(node_size, node_size, bias=False)
        self.combined_mlp = nn.Sequential(
            nn.Linear(2 * node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(self, node_state_scalar, node_state_vector):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """

        Uv = self.linearU(node_state_vector)  # num_nodes, 3, node_size
        Vv = self.linearV(node_state_vector)  # num_nodes, 3, node_size

        Vv_norm = torch.linalg.norm(Vv, dim=1, keepdim=False)  # num_nodes, node_size

        mlp_input = torch.cat(
            (node_state_scalar, Vv_norm), dim=1
        )  # num_nodes, node_size*2
        mlp_output = self.combined_mlp(mlp_input)

        a_ss, a_sv, a_vv = torch.split(
            mlp_output, node_state_scalar.shape[1], dim=1
        )  # num_nodes, node_size

        inner_prod = torch.sum(Uv * Vv, dim=1)  # num_nodes, node_size

        delta_v = torch.unsqueeze(a_vv, 1) * Uv  # num_nodes, 3, node_size

        delta_s = a_ss + a_sv * inner_prod  # num_nodes, node_size

        return node_state_scalar + delta_s, node_state_vector + delta_v


class EnrichedRadialFilter(nn.Module):
    """Radial filter with a multi-scale enriched basis and a nonlinear MLP.

    Augments the fixed sinc basis (edge_state) with trainable-frequency
    Bessel functions sin(w_n * d) / (d + 1e-5) [DimeNet, Klicpera et al.,
    ICLR 2020], smooth inverse-distance powers (1 / (d + 1))^k and Gaussian
    channels, then maps the enriched basis through a SiLU MLP [NequIP,
    Batzner et al., Nat. Commun. 2022]. All channels are functions of the
    scalar distance only, so full O(3) invariance is preserved; every
    function is C^2-smooth for d >= 0 (Bessel at d->0 tends to w_n, and
    the 1e-5 regularizer keeps it finite).
    """

    def __init__(
        self,
        edge_size,
        node_size,
        cutoff,
        n_bessel=16,
        n_invdist=4,
        n_gauss=8,
        hidden=64,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.n_bessel = n_bessel
        self.n_invdist = n_invdist
        self.n_gauss = n_gauss

        bessel_w = torch.linspace(1.0, float(n_bessel), n_bessel) * (math.pi / cutoff)
        self.bessel_freq = nn.Parameter(bessel_w)
        self.gauss_mean = nn.Parameter(torch.linspace(0.3, 3.8, n_gauss))
        self.gauss_log_sigma = nn.Parameter(
            torch.full((n_gauss,), math.log(0.6))
        )
        self.register_buffer(
            "inv_powers", torch.arange(1, n_invdist + 1).float()
        )

        self.mlp = nn.Sequential(
            nn.Linear(edge_size + n_bessel + n_invdist + n_gauss, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * node_size),
        )

    def forward(self, edge_state, edge_distance):
        """Enriched-basis filter weights.

        Args:
            edge_state (tensor): Sinc basis features (num_edges, edge_size)
            edge_distance (tensor): Edge distances (num_edges, 1)

        Returns:
            filter_weight (num_edges, 3 * node_size) (envelope NOT applied)
        """
        d = edge_distance  # (E, 1)
        bessel = torch.sin(d * self.bessel_freq) / (d + 1e-5)  # (E, n_bessel)
        invdist = torch.pow(1.0 / (d + 1.0), self.inv_powers)  # (E, n_invdist)
        gauss = torch.exp(
            -torch.square((d - self.gauss_mean) / torch.exp(self.gauss_log_sigma))
        )  # (E, n_gauss)
        enriched = torch.cat([edge_state, bessel, invdist, gauss], dim=1)
        return self.mlp(enriched)


class PaiNNInteraction(nn.Module):
    """Interaction network"""

    def __init__(self, node_size, edge_size, cutoff):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
            cutoff (float): Cutoff distance
        """
        super().__init__()

        self.filter_layer = EnrichedRadialFilter(edge_size, node_size, cutoff)

        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

    def forward(
        self,
        node_state_scalar,
        node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        """
        Args:
            node_state_scalar (tensor): Node states (num_nodes, node_size)
            node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)
            edge_vector (tensor): Edge vector difference between nodes (num_edges, 3)
            edge_distance (tensor): l2-norm of edge_vector (num_edges, 1)
            edges (tensor): Directed edges with node indices (num_edges, 2)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        # Compute all messages
        edge_vector_normalised = edge_vector / torch.maximum(
            torch.linalg.norm(edge_vector, dim=1, keepdim=True), torch.tensor(1e-12)
        )  # num_edges, 3

        filter_weight = self.filter_layer(
            edge_state, edge_distance
        )  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

        scalar_output = self.scalar_message_mlp(
            node_state_scalar
        )  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = torch.split(
            filter_output, node_state_scalar.shape[1], dim=1
        )

        gate_state_vector = torch.unsqueeze(
            gate_state_vector, 1
        )  # num_edges, 1, node_size
        gate_edge_vector = torch.unsqueeze(
            gate_edge_vector, 1
        )  # num_edges, 1, node_size

        # Only include sender in messages
        messages_scalar = node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * torch.unsqueeze(
            edge_vector_normalised, 2
        )

        # Sum messages
        message_sum_scalar = torch.zeros_like(node_state_scalar)
        message_sum_scalar.index_add_(0, edges[:, 1], messages_scalar)
        message_sum_vector = torch.zeros_like(node_state_vector)
        message_sum_vector.index_add_(0, edges[:, 1], messages_state_vector)

        # State transition
        new_state_scalar = node_state_scalar + message_sum_scalar
        new_state_vector = node_state_vector + message_sum_vector

        return new_state_scalar, new_state_vector


class PaiNNInteractionOneWay(nn.Module):
    """Sasme as Interaction network, but the receiving nodes are differently indexed from the sending nodes"""

    def __init__(self, node_size, edge_size, cutoff):
        """
        Args:
            node_size (int): Size of node state
            edge_size (int): Size of edge state
            cutoff (float): Cutoff distance
        """
        super().__init__()

        self.filter_layer = EnrichedRadialFilter(edge_size, node_size, cutoff)

        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, 3 * node_size),
        )

        # Ignore messages gate (not part of original PaiNN network).
        # State-aware: conditioned on message sum, current receiver state and
        # the invariant per-channel magnitude of the vector message sum.
        self.update_gate_mlp = nn.Sequential(
            nn.Linear(3 * node_size, 2 * node_size),
            nn.SiLU(),
            nn.Linear(2 * node_size, 2 * node_size),
            nn.Sigmoid(),
        )

        # Direct atom->probe multipole projection: low-dim (D=32) learned
        # projections for the O(3)-invariant geometric contractions along each
        # edge: <v_j, rhat>, <v_p, rhat>, <v_j, v_p>, <U v_j, rhat>^2.
        # D is deliberately small: the per-edge (E, F)-sized intermediates at
        # val-scale edge counts (~760k) would otherwise fragment the CUDA
        # allocator and blow past the 16 GB VRAM ceiling (WSL2 then pages to
        # host RAM and the forward slows down by >10x).
        self.direction_dim = 32
        self.geo_proj_sender = nn.Linear(node_size, self.direction_dim, bias=False)
        self.geo_proj_receiver = nn.Linear(node_size, self.direction_dim, bias=False)
        self.geo_proj_quad = nn.Linear(node_size, self.direction_dim, bias=False)

        # Gate built from the four contraction channels. Output is a
        # per-edge, per-channel multiplicative factor (1 + tanh(...)) in
        # (0, 2). With a zero-initialized last layer the factor is exactly
        # 1.0 at random init (model == canonical PaiNN one-way interaction)
        # while gradients still flow (tanh'(0) = 1).
        self.directional_gate_mlp = nn.Sequential(
            nn.Linear(4 * self.direction_dim, 2 * self.direction_dim),
            nn.SiLU(),
            nn.Linear(2 * self.direction_dim, node_size),
            nn.Tanh(),
        )
        nn.init.zeros_(self.directional_gate_mlp[2].weight)
        nn.init.zeros_(self.directional_gate_mlp[2].bias)
        # ReZero-style zero init: at random init the model behaves exactly
        # like the canonical PaiNN one-way interaction (dir gate factor = 1).
        nn.init.zeros_(self.directional_gate_mlp[2].weight)
        nn.init.zeros_(self.directional_gate_mlp[2].bias)

    def forward(
        self,
        sender_node_state_scalar,
        sender_node_state_vector,
        receiver_node_state_scalar,
        receiver_node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        """
        Args:
            sender_node_state_scalar (tensor): Node states (num_nodes, node_size)
            sender_node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            receiver_node_state_scalar (tensor): Node states (num_nodes, node_size)
            receiver_node_state_vector (tensor): Node states (num_nodes, 3, node_size)
            edge_state (tensor): Edge states (num_edges, edge_size)
            edge_vector (tensor): Edge vector difference between nodes (num_edges, 3)
            edge_distance (tensor): l2-norm of edge_vector (num_edges, 1)
            edges (tensor): Directed edges with node indices (num_edges, 2)

        Returns:
            Tuple of 2 tensors:
                updated_node_state_scalar (num_nodes, node_size)
                updated_node_state_vector (num_nodes, 3, node_size)
        """
        # Compute all messages
        edge_vector_normalised = edge_vector / torch.maximum(
            torch.linalg.norm(edge_vector, dim=1, keepdim=True), torch.tensor(1e-12)
        )  # num_edges, 3

        scalar_output = self.scalar_message_mlp(
            sender_node_state_scalar
        )  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size

        # Sum messages. The whole per-edge block (including the enriched
        # radial filter, which has its own MLP) is processed in chunks: at
        # val-scale edge counts (~760k) full-size per-edge intermediates
        # ((E, 3, F) tensors are ~1.2 GB each) fragment the CUDA allocator and
        # push reserved memory past the 16 GB VRAM ceiling, after which WSL2
        # pages to host RAM and the forward slows down by an order of
        # magnitude. Chunking keeps live per-edge memory at ~0.1 GB while
        # index_add_ accumulation preserves the exact same graph/result.
        message_sum_scalar = torch.zeros_like(receiver_node_state_scalar)
        message_sum_vector = torch.zeros_like(receiver_node_state_vector)

        rhat = edge_vector_normalised  # (E, 3)
        edge_chunk = 65536
        for st in range(0, edges.shape[0], edge_chunk):
            en = min(st + edge_chunk, edges.shape[0])
            e = edges[st:en]
            r_chunk = rhat[st:en]
            rhat_u = torch.unsqueeze(r_chunk, -1)  # (ch, 3, 1)

            # Enriched radial filter, computed per chunk (memory-safe).
            filter_weight = self.filter_layer(
                edge_state[st:en], edge_distance[st:en]
            )  # (ch, 3*node_size)
            filter_weight = filter_weight * cosine_cutoff(
                edge_distance[st:en], self.cutoff
            )
            filter_output = filter_weight * scalar_output[st:en]

            gate_state_vector, gate_edge_vector, gate_node_state = torch.split(
                filter_output, sender_node_state_scalar.shape[1], dim=1
            )
            gate_state_vector = torch.unsqueeze(
                gate_state_vector, 1
            )  # (ch, 1, node_size)
            gate_edge_vector = torch.unsqueeze(
                gate_edge_vector, 1
            )  # (ch, 1, node_size)

            # Direct atom->probe multipole projections: O(3)-invariant,
            # parity-even contractions (polar-vector inner products and
            # squares only), giving the probe scalar state first-hop
            # directional information about the atom's equivariant multipoles
            # (fixes layer-0 directional blindness).
            vec_sender = sender_node_state_vector[e[:, 0]]  # (ch, 3, F)
            vec_receiver = receiver_node_state_vector[e[:, 1]]  # (ch, 3, F)
            # NOTE: explicit mul+sum instead of einsum — torch.einsum is ~40x
            # slower for these small-index contractions.
            v_s_proj = self.geo_proj_sender(vec_sender)  # (ch, 3, D)
            v_r_proj = self.geo_proj_receiver(vec_receiver)  # (ch, 3, D)
            proj_sender = torch.sum(v_s_proj * rhat_u, dim=1)  # <v_j, rhat>
            proj_receiver = torch.sum(v_r_proj * rhat_u, dim=1)  # <v_p, rhat>
            inner_sr = torch.sum(v_s_proj * v_r_proj, dim=1)  # <v_j, v_p>
            quad_proj = torch.sum(self.geo_proj_quad(vec_sender) * rhat_u, dim=1)
            quad_proj.square_()  # <U v_j, rhat>^2 (parity-even)
            dir_features = torch.cat(
                [proj_sender, proj_receiver, inner_sr, quad_proj], dim=1
            )  # (ch, 4D)
            dir_gate = self.directional_gate_mlp(dir_features)  # (ch, F)

            # Only include sender in messages. The same directional gate
            # (1 + dir_gate) that modulates the scalar message also gates the
            # sender-vector contribution, giving the vector channel the same
            # O(3)-invariant directional selectivity (parity-even factor).
            messages_scalar = (
                sender_node_state_scalar[e[:, 0]] * gate_node_state * (1.0 + dir_gate)
            )
            messages_state_vector = (
                vec_sender * gate_state_vector * (1.0 + dir_gate).unsqueeze(1)
                + gate_edge_vector * torch.unsqueeze(r_chunk, 2)
            )

            message_sum_scalar.index_add_(0, e[:, 1], messages_scalar)
            message_sum_vector.index_add_(0, e[:, 1], messages_state_vector)

        # State transition: condition the gates on the message sum, the current
        # probe state, and the invariant per-channel vector message magnitude.
        message_sum_vector_norm = torch.linalg.norm(message_sum_vector, dim=1)
        gate_input = torch.cat(
            [message_sum_scalar, receiver_node_state_scalar, message_sum_vector_norm],
            dim=1,
        )
        update_gate_scalar, update_gate_vector = torch.split(
            self.update_gate_mlp(gate_input),
            receiver_node_state_scalar.shape[1],
            dim=1,
        )
        update_gate_vector = torch.unsqueeze(
            update_gate_vector, 1
        )  # num_nodes, 1, node_size
        new_state_scalar = (
            update_gate_scalar * receiver_node_state_scalar
            + (1.0 - update_gate_scalar) * message_sum_scalar
        )
        new_state_vector = (
            update_gate_vector * receiver_node_state_vector
            + (1.0 - update_gate_vector) * message_sum_vector
        )

        return new_state_scalar, new_state_vector


def sinc_expansion(input_x: torch.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a sinc-like basis function expansion.
    Based on [1].
    sin(n*pi*f/rcut)/f

    [1] arXiv:2003.03123 - Directional Message Passing for Molecular Graphs

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (n, cutoff) tuples

    Return:
        (num_edges, n1+n2+...) tensor
    """
    feat_list = torch.unbind(input_x, dim=1)
    expanded_list = []
    for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
        assert feat is not None, "Too many expansion parameters given"
        if step_tuple:
            n, cutoff = step_tuple
            feat_expanded = torch.unsqueeze(feat, dim=1)
            n_range = torch.arange(n, device=input_x.device, dtype=input_x.dtype) + 1
            # multiplication by pi n_range / cutoff is done in original painn for some reason
            out = torch.sinc(n_range/cutoff*feat_expanded)*np.pi*n_range/cutoff
            expanded_list.append(out)
        else:
            expanded_list.append(torch.unsqueeze(feat, 1))
    return torch.cat(expanded_list, dim=1)


def cosine_cutoff(distance: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        distance < cutoff,
        0.5 * (torch.cos(np.pi * distance / cutoff) + 1),
        torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
    )
