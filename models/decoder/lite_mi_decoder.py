# models/decoder/lite_mi_decoder.py
# ------------------------------------------------------------------------
# Motion-aware multi-expert decoder layer for MOTRv2/MRTrack.
# ------------------------------------------------------------------------

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.ops.modules import MSDeformAttn


def _get_activation_fn(activation: str):
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class ReLUDropout(nn.Dropout):
    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training, inplace=self.inplace)


def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)
    mask = (x < 0) | (torch.rand_like(x) > 1 - p)
    return x.masked_fill_(mask, 0).div_(1 - p) if inplace else x.masked_fill(mask, 0).div(1 - p)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = _get_activation_fn(activation)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.fc2(self.drop1(self.act(self.fc1(x))))
        x = self.norm(x + self.drop2(y))
        return x


class LiteMIDecoderLayer(nn.Module):
    """Shared self-attention plus B deformable cross-attention experts.

    The fusion switch mirrors Table III:
      sum     - element-wise summation of expert outputs
      linear  - concat-linear fusion
      hard    - top-1 router gate
      softmax - query-adaptive softmax gate, the default MAE variant
    """

    _VALID_FUSIONS = {"sum", "linear", "hard", "softmax"}

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        sigmoid_attn: bool = False,
        num_branches: int = 2,
        fusion: str = "softmax",
        router_tau: float = 1.0,
    ):
        super().__init__()
        if fusion not in self._VALID_FUSIONS:
            raise ValueError(f"Bad MAE fusion '{fusion}'. Expected one of {sorted(self._VALID_FUSIONS)}.")

        self.num_branches = num_branches
        self.fusion = fusion
        self.router_tau = float(router_tau)
        self.last_gates = None

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(d_model)

        branches = []
        for _ in range(num_branches):
            branches.append(
                nn.ModuleDict(
                    {
                        "cross": MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn),
                        "cross_drop": nn.Dropout(dropout),
                        "cross_norm": nn.LayerNorm(d_model),
                        "ffn": FeedForward(d_model, d_ffn, dropout, activation),
                    }
                )
            )
        self.branches = nn.ModuleList(branches)

        self.router = nn.Linear(d_model, num_branches)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.merge = nn.Linear(num_branches * d_model, d_model)
        self.dropout_merge = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self._init_merge_as_average()

    def _init_merge_as_average(self):
        with torch.no_grad():
            w = self.merge.weight
            b = self.merge.bias
            w.zero_()
            d = self.merge.out_features
            eye = torch.eye(d, device=w.device, dtype=w.dtype)
            for k in range(self.num_branches):
                w[:, k * d : (k + 1) * d].copy_(eye / self.num_branches)
            b.zero_()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def _self_attend(
        self,
        tgt: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        if attn_mask is not None:
            sa_out = self.self_attn(q, k, v, attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            sa_out = self.self_attn(q, k, v)[0].transpose(0, 1)
        return self.norm_sa(tgt + self.dropout_sa(sa_out))

    def _router_gates(self, router_in: torch.Tensor):
        logits = self.router(router_in) / max(self.router_tau, 1e-6)
        if self.fusion == "hard":
            soft_gates = F.softmax(logits, dim=-1)
            top1 = soft_gates.argmax(dim=-1)
            hard_gates = F.one_hot(top1, num_classes=self.num_branches).to(dtype=router_in.dtype)
            gates = hard_gates + soft_gates - soft_gates.detach()
            self.last_gates = hard_gates.detach()
            return gates
        else:
            gates = F.softmax(logits, dim=-1)
        self.last_gates = gates.detach()
        return gates

    def _fuse(self, branch_feats: List[torch.Tensor], router_in: torch.Tensor):
        feats = torch.stack(branch_feats, dim=2)  # [B, Nq, M, C]
        if self.fusion == "sum":
            self.last_gates = None
            return feats.sum(dim=2)
        if self.fusion == "linear":
            self.last_gates = None
            return self.merge(feats.reshape(feats.shape[0], feats.shape[1], -1))

        gates = self._router_gates(router_in)
        return (feats * gates.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        reference_points_input: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        src_level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        mem_bank=None,
        mem_bank_pad_mask=None,
        attn_mask: Optional[torch.Tensor] = None,
        rp_input_per_branch: Optional[List[torch.Tensor]] = None,
    ):
        h = self._self_attend(tgt, query_pos, attn_mask)
        router_in = self.with_pos_embed(h, query_pos)

        branch_feats = []
        for k, branch in enumerate(self.branches):
            rpi_k = rp_input_per_branch[k] if rp_input_per_branch is not None else reference_points_input
            z = branch["cross"](
                router_in,
                rpi_k,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )
            z = branch["cross_norm"](h + branch["cross_drop"](z))
            branch_feats.append(branch["ffn"](z))

        merged = self._fuse(branch_feats, router_in)
        return self.norm_out(tgt + self.dropout_merge(merged))
