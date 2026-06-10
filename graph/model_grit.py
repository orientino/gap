"""
GRIT: Graph Inductive Bias Transformer for graph regression (ZINC).
Standalone PyG implementation following the original code:
  https://github.com/LiamMa/GRIT
  https://arxiv.org/pdf/2305.17589

dependencies:
    - opt_einsum
    - torch_sparse
"""

import warnings

import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_sparse
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max
from torch_sparse import SparseTensor

# ZINC: 28 atom types (node features), 4 bond types (edge features)
N_ATOM_TYPES = 28
N_BOND_TYPES = 4


# =========================================================================
# RRWP Transform
# (from grit/transform/rrwp.py)
# =========================================================================


def full_edge_index(edge_index, batch=None):
    """
    Return the full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        adj = torch.ones((n, n), dtype=torch.short, device=edge_index.device)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        negative_index_list.append(_edge_index + cum_nodes[i])

    return torch.cat(negative_index_list, dim=1).contiguous()


@torch.no_grad()
def add_full_rrwp(
    data,
    walk_length=8,
    attr_name_abs="rrwp",
    attr_name_rel="rrwp",
    add_identity=True,
    spd=False,
    **kwargs,
):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", None)

    adj = SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes)
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x walk_length

    abs_pe = pe.diagonal().transpose(0, 1)  # n x walk_length

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # switch row/col: framework performs right-mul while adj is row-normalized
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    if spd:
        spd_idx = walk_length - torch.arange(walk_length, device=device)
        val = (rel_pe_val > 0).float() * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).float()
        abs_pe = torch.zeros_like(abs_pe)

    data[attr_name_abs] = abs_pe
    data[f"{attr_name_rel}_index"] = rel_pe_idx
    data[f"{attr_name_rel}_val"] = rel_pe_val
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.long()

    return data


class AddFullRRWP(pyg.transforms.BaseTransform):
    """PyG transform wrapper for add_full_rrwp."""

    def __init__(self, walk_length=8, **kwargs):
        self.walk_length = walk_length
        self.kwargs = kwargs

    def forward(self, data):
        return add_full_rrwp(data, walk_length=self.walk_length, **self.kwargs)


# =========================================================================
# RRWP Encoders
# (from grit/encoder/rrwp_encoder.py)
# =========================================================================


class RRWPLinearNodeEncoder(nn.Module):
    """
    FC_1(RRWP) + FC_2 (Node-attr)
    note: FC_2 is given by the Typedict encoder of node-attr in some cases
    Parameters:
    num_classes - the number of classes for the embedding mapping to learn
    """

    def __init__(
        self,
        emb_dim,
        out_dim,
        use_bias=False,
        batchnorm=False,
        layernorm=False,
        pe_name="rrwp",
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if self.batchnorm:
            rrwp = self.bn(rrwp)
        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp.to(batch.x.dtype)
        else:
            batch.x = rrwp

        return batch


class RRWPLinearEdgeEncoder(nn.Module):
    """
    Merge RRWP with given edge-attr and Zero-padding to all pairs of node.
    FC_1(RRWP) + FC_2(edge-attr)
    - FC_2 given by the TypedictEncoder in some cases
    - Zero-padding for non-existing edges in fully-connected graph
    - (optional) add node-attr as the E_{i,i}'s attr
        note: assuming node-attr and edge-attr is with the same dimension after Encoders
    """

    def __init__(
        self,
        emb_dim,
        out_dim,
        batchnorm=False,
        layernorm=False,
        use_bias=False,
        pad_to_full_graph=True,
        fill_value=0.0,
        add_node_attr_as_self_loop=False,
        overwrite_old_attr=False,
    ):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on "
                "providing shortest-path distance info "
            )

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.0

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            # edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.0
            )

            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val.to(edge_attr.dtype)], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes, op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)
        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(pad_to_full_graph={self.pad_to_full_graph},"
            f"fill_value={self.fill_value},"
            f"{self.fc.__repr__()})"
        )


# =========================================================================
# Attention and Transformer Layer
# (from grit/layer/grit_layer.py)
# =========================================================================


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn(
            "Compute the degree on the fly; Might be problematic if have "
            "applied edge-padding to complete graphs"
        )
        deg = pyg.utils.degree(
            batch.edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float
        )
        log_deg = torch.log(deg + 1)
    return log_deg.view(batch.num_nodes, 1)


act_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
    Proposed Attention Computation for GRIT
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        use_bias,
        clamp=5.0,
        dropout=0.0,
        act=None,
        edge_enhance=True,
        sqrt_relu=False,
        signed_sqrt=True,
        **kwargs,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(
            torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True
        )
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num relative) x num_heads x out_dim
        score = src + dest  # element-wise addition

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, : self.out_dim], batch.E[:, :, self.out_dim :]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(
            score, batch.edge_index[1]
        )  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = (
            batch.V_h[batch.edge_index[0]] * score
        )  # (num relative) x num_heads x out_dim
        batch.wV = scatter(
            msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="add"
        )

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get("wE", None)

        return h_out, e_out


class GritTransformerLayer(nn.Module):
    """
    Proposed Transformer Layer for GRIT
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        act="relu",
        norm_e=True,
        O_e=True,
        update_e=True,
        bn_momentum=0.1,
        bn_no_runner=False,
        rezero=False,
        deg_scaler=True,
        attn_use_bias=False,
        attn_clamp=5.0,
        attn_act="relu",
        attn_edge_enhance=True,
        attn_sqrt_relu=False,
        attn_signed_sqrt=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.update_e = update_e
        self.bn_momentum = bn_momentum
        self.bn_no_runner = bn_no_runner
        self.rezero = rezero

        self.act = act_dict[act]() if act is not None else nn.Identity()
        self.deg_scaler = deg_scaler

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=attn_use_bias,
            dropout=attn_dropout,
            clamp=attn_clamp,
            act=attn_act,
            edge_enhance=attn_edge_enhance,
            sqrt_relu=attn_sqrt_relu,
            signed_sqrt=attn_signed_sqrt,
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(
                torch.zeros(1, out_dim // num_heads * num_heads, 2)
            )
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch
            # leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not bn_no_runner,
                eps=1e-5,
                momentum=bn_momentum,
            )
            self.batch_norm1_e = (
                nn.BatchNorm1d(
                    out_dim,
                    track_running_stats=not bn_no_runner,
                    eps=1e-5,
                    momentum=bn_momentum,
                )
                if norm_e
                else nn.Identity()
            )

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not bn_no_runner,
                eps=1e-5,
                momentum=bn_momentum,
            )

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg.to(h.dtype)], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return (
            "{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]".format(
                self.__class__.__name__,
                self.in_channels,
                self.out_channels,
                self.num_heads,
                self.residual,
                super().__repr__(),
            )
        )


# =========================================================================
# GRIT Model (ZINC graph regression)
# =========================================================================


class GRIT(nn.Module):
    """
    Graph Inductive Bias Transformer (GRIT) for graph-level regression on ZINC.

    Requires data pre-processed with AddFullRRWP(walk_length=ksteps), which adds
    batch.rrwp, batch.rrwp_index, batch.rrwp_val, batch.log_deg, batch.deg.
    """

    def __init__(
        self,
        n_layers=10,
        d_embed=64,
        n_heads=8,
        ksteps=21,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        norm_e=True,
        O_e=True,
        update_e=True,
        bn_momentum=0.1,
        deg_scaler=True,
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(N_ATOM_TYPES, d_embed)
        self.bond_embed = nn.Embedding(N_BOND_TYPES, d_embed)
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, d_embed)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(
            ksteps,
            d_embed,
            pad_to_full_graph=True,
            add_node_attr_as_self_loop=False,
            fill_value=0.0,
        )
        self.layers = nn.ModuleList(
            [
                GritTransformerLayer(
                    in_dim=d_embed,
                    out_dim=d_embed,
                    num_heads=n_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    residual=residual,
                    norm_e=norm_e,
                    O_e=O_e,
                    update_e=update_e,
                    bn_momentum=bn_momentum,
                    deg_scaler=deg_scaler,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_embed, 1)

    def forward(self, data):
        data.x = self.atom_embed(data.x.squeeze(-1))
        data.edge_attr = self.bond_embed(data.edge_attr.squeeze(-1))
        data = self.rrwp_abs_encoder(data)
        data = self.rrwp_rel_encoder(data)
        for layer in self.layers:
            data = layer(data)
        h = global_add_pool(data.x, data.batch)
        return self.head(h).squeeze(-1)


if __name__ == "__main__":
    import torch
    from torch_geometric.data import Batch

    ksteps = 21
    transform = AddFullRRWP(walk_length=ksteps)

    # Minimal synthetic graph batch
    graphs = []
    for _ in range(2):
        n = 10
        x = torch.randint(0, N_ATOM_TYPES, (n, 1))
        edge_index = torch.randint(0, n, (2, 20))
        edge_attr = torch.randint(0, N_BOND_TYPES, (20, 1))
        y = torch.randn(1)
        from torch_geometric.data import Data

        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        g = transform(g)
        graphs.append(g)

    data = Batch.from_data_list(graphs)

    model = GRIT(n_layers=10, d_embed=64, n_heads=8, ksteps=ksteps)  # 0.47M
    model = GRIT(n_layers=6, d_embed=64, n_heads=6, ksteps=ksteps)  # 0.27M
    out = model(data)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"GRIT: output {out.shape}, params {params:.2f}M")
