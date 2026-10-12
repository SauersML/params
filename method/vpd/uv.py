"""VPD's subcomponents as vectors (#2951): (site, index) -> its read vector v [d_in], write vector u [d_out] and scale
|u||v|, from the s-55ea3f9b decomposition of vpd4l.

A weight matrix W [d_out, d_in] is sum_c u_c v_c^T plus its remainder; subcomponent c computes a_c = v_c . x on the
matrix's input x and writes a_c u_c. The pair is defined up to (u, v) -> (s u, v / s): the loader gives the unit
directions and the product of the norms, and orients them (u v^T = (-u)(-v)^T) so that the activation is positive
where the subcomponent typically acts: the sign of its median activation at its top contexts (the atlas's lens_sign),
or without an atlas the sign that makes the largest read entry positive (part_tokens.py's convention).

The spaces (vpd4l: d = 768, 6 heads of 128, MLP width 3,072):
  q_proj, k_proj  v: the normed residual stream entering the layer (rms(x) * g_1) [768]; u: query or key space before
                  rotary embedding, the 6 heads' 128 dimensions concatenated [768]
  v_proj          v: the same normed residual [768]; u: value space, heads concatenated [768]
  o_proj          v: the attention output, each head's weighted values concatenated [768]; u: the residual stream [768]
  c_fc            v: the normed residual entering the MLP (rms(x) * g_2) [768]; u: the MLP's pre-activation [3072]
  down_proj       v: the MLP's activation after GELU [3072]; u: the residual stream [768]

The tensors are ~/mpd-data/oracle/vpd/uv.safetensors ({site}.U [C, d_out], {site}.V [d_in, C]; the checkpoint's own
values, published in rp-run-data for pods).

  python -m method.vpd.uv L2.o#77          one subcomponent's norms, orientation and space
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

PATH = Path.home() / "mpd-data/oracle/vpd/uv.safetensors"
ATLAS = Path.home() / "mpd-data/atlas/vpd4l"
SPACES = {"q_proj": ("normed residual (attention input)", "query space, heads concatenated, before rotary"),
          "k_proj": ("normed residual (attention input)", "key space, heads concatenated, before rotary"),
          "v_proj": ("normed residual (attention input)", "value space, heads concatenated"),
          "o_proj": ("attention output, heads concatenated", "residual stream"),
          "c_fc": ("normed residual (MLP input)", "MLP pre-activation"),
          "down_proj": ("MLP activation after GELU", "residual stream")}
KINDS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "fc": "c_fc", "down": "down_proj"}


def site_name(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"


class UV:
    """The subcomponents' vectors, loaded once (map_location cpu)."""

    def __init__(self, path: Path = PATH, atlas: Path | None = ATLAS):
        from safetensors.torch import load_file

        self.t = load_file(str(path))
        self.sign = None
        if atlas is not None and (Path(atlas) / "atlas.safetensors").exists():
            import json

            from safetensors import safe_open

            meta = json.loads((Path(atlas) / "atlas.json").read_text())
            with safe_open(str(Path(atlas) / "atlas.safetensors"), "pt") as f:
                flat = f.get_tensor("lens_sign").float()
            off, s = {}, 0
            for n in meta["names"]:
                off[n] = (s, meta["counts"][n])
                s += meta["counts"][n]
            self.sign = {n: flat[a:a + c] for n, (a, c) in off.items()}

    def names(self) -> list[str]:
        return sorted({k.rsplit(".", 1)[0] for k in self.t})

    def count(self, site: str) -> int:
        return self.t[f"{site}.U"].shape[0]

    def raw(self, site: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(v [d_in], u [d_out]) as the checkpoint holds them."""
        return self.t[f"{site}.V"][:, index], self.t[f"{site}.U"][index]

    def get(self, site: str, index: int) -> dict:
        """{"v": unit read direction [d_in], "u": unit write direction [d_out], "scale": |u||v|, "sign": the orientation
        applied (+1 or -1), "spaces": (read space, write space)}; u v^T scale = the checkpoint's slice."""
        v, u = self.raw(site, index)
        nv, nu = float(v.norm()), float(u.norm())
        if self.sign is not None:
            sign = float(self.sign[site][index]) or 1.0
        else:
            sign = float(torch.sign(v[v.abs().argmax()])) or 1.0
        return {"v": sign * v / max(nv, 1e-30), "u": sign * u / max(nu, 1e-30), "scale": nu * nv, "sign": sign,
                "spaces": SPACES[site.rsplit(".", 1)[1]]}

    def slice(self, site: str, index: int) -> torch.Tensor:
        """The subcomponent's rank-one part of W, u v^T [d_out, d_in]."""
        v, u = self.raw(site, index)
        return torch.outer(u, v)


def parse(name: str) -> tuple[str, int]:
    """L<layer>.<kind>#<index> -> (site, index)."""
    m = re.fullmatch(r"L(\d+)\.(q|k|v|o|fc|down)#(\d+)", name.strip())
    if not m:
        raise ValueError(name)
    return site_name(int(m[1]), KINDS[m[2]]), int(m[3])


if __name__ == "__main__":
    site, index = parse(sys.argv[1])
    e = UV().get(site, index)
    print(f"{site} #{index}: reads {e['spaces'][0]} [{e['v'].numel()}], writes {e['spaces'][1]} [{e['u'].numel()}], "
          f"|u||v| {e['scale']:.4f}, orientation {e['sign']:+.0f}")
