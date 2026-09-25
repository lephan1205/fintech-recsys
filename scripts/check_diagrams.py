"""Acceptance checklist for ``docs/diagrams/*.svg`` (run by ``tests/test_docs.py``).

Automated checks:

1. Every learned component (``nn.Module`` subclasses under ``src/recsys/models`` and
   ``src/recsys/losses``) and every public loss function (``*_loss`` in ``losses/``) is
   referenced as ``path.py::Name`` in at least one diagram.
2. Every ``path.py::Name[.member]`` reference in either diagram resolves to an existing
   module, top-level symbol and (when given) attribute — no box names a component that
   does not exist.
3. Within one diagram, no two boxes carry the same implementing-symbol line (each
   component is drawn once per diagram).
4. Every hyper-parameter printed in a diagram equals the config default it describes.
5. Each file is <= 150 KB, 900 px wide, and every text element is >= 11 px.
6. Both diagrams carry the legend conventions and the serving diagram marks the three
   eligibility enforcement points.

Edge shapes vs tensor comments and "opens on GitHub" are checked by eye.
"""

from __future__ import annotations

import ast
import html
import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data.schema import PRODUCT_FEATURE_DIM, USER_FEATURE_DIM, DelayConfig  # noqa: E402
from recsys.layers.rq_vae import RQVAEConfig  # noqa: E402
from recsys.models.prm.model import PRMConfig  # noqa: E402
from recsys.training.config import TrainingConfig  # noqa: E402

DIAGRAMS = ROOT / "docs" / "diagrams"
SRC = ROOT / "src" / "recsys"
MAX_BYTES = 150 * 1024
MIN_FONT_PX = 11.0
WIDTH = 900

_REF = re.compile(r"([A-Za-z_][\w/]*\.py)::([A-Za-z_][\w.]*)")
_REF_TAIL = re.compile(r"\s*,\s*([A-Za-z_][\w.]*)(?=\s*(?:,|$|[^\w.]))")


def texts(svg: str) -> list[tuple[str, bool]]:
    """(text, is_symbol_line) for every ``<text>`` element."""
    out = []
    for m in re.finditer(r"<text([^>]*)>([^<]*)</text>", svg):
        attrs, body = m.group(1), html.unescape(m.group(2))
        out.append((body, "font-style='italic'" in attrs))
    return out


def references(line: str) -> list[tuple[str, str]]:
    """``a/b.py::X.m, Y`` -> [(a/b.py, X.m), (a/b.py, Y)] for every ref in the line."""
    refs: list[tuple[str, str]] = []
    for m in _REF.finditer(line):
        path = m.group(1)
        refs.append((path, m.group(2)))
        tail = line[m.end() :]
        while (t := _REF_TAIL.match(tail)) is not None:
            refs.append((path, t.group(1)))
            tail = tail[t.end() :]
    return refs


def required_components() -> dict[str, str]:
    """Class / function name -> defining path (relative to src/recsys)."""
    comps: dict[str, str] = {}
    for folder in ("models", "losses"):
        for f in sorted((SRC / folder).rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            rel = f.relative_to(SRC).as_posix()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = {ast.unparse(b) for b in node.bases}
                    if bases & {"nn.Module", "torch.nn.Module"}:
                        comps[node.name] = rel
                elif (
                    isinstance(node, ast.FunctionDef)
                    and folder == "losses"
                    and node.name.endswith("_loss")
                    and not node.name.startswith("_")
                ):
                    comps[node.name] = rel
    return comps


def resolve(path: str, symbol: str) -> str | None:
    module_path = SRC / path
    if not module_path.exists():
        return f"{path} does not exist"
    mod = importlib.import_module("recsys." + path[:-3].replace("/", "."))
    head, *rest = symbol.split(".")
    if not hasattr(mod, head):
        return f"{path} has no top-level symbol {head}"
    obj = getattr(mod, head)
    for attr in rest:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
            continue
        # instance attributes (``self.towers = ...``) are found in the class source
        if inspect.isclass(obj) and re.search(rf"self\.{attr}\s*[:=]", inspect.getsource(obj)):
            return None
        return f"{path}::{head} has no attribute {attr}"
    return None


def hyperparameter_strings() -> dict[str, list[str]]:
    """Diagram file -> substrings that must appear, formatted from the config defaults."""
    cfg = TrainingConfig()
    m = cfg.models
    rq = RQVAEConfig(input_dim=PRODUCT_FEATURE_DIM)
    prm = PRMConfig(cand_dim=m.d_model, user_dim=m.d_model)
    fusion = 3 * m.d_model + USER_FEATURE_DIM + PRODUCT_FEATURE_DIM
    ro, to, po, qo = (
        cfg.ranker_optimizer,
        cfg.tiger_optimizer,
        cfg.prm_optimizer,
        cfg.rqvae_optimizer,
    )  # noqa: E501
    fl, ub, ut, cal = cfg.funnel_loss, cfg.user_benefit, cfg.utility, cfg.calibration

    def pct(x: float) -> str:
        return f"{x * 100:g} %"

    def sci(x: float) -> str:  # 0.001 -> 1e-3
        return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")

    def thousands(n: int) -> str:  # 2000 -> 2 000
        return f"{n:,}".replace(",", " ")

    serving = [
        f"RQ-VAE {PRODUCT_FEATURE_DIM}→{m.rq_hidden}→{m.rq_latent_dim}→{m.rq_hidden}→{PRODUCT_FEATURE_DIM}",  # noqa: E501
        f"depth D = {m.rq_levels}",
        f"depth D = {m.rq_levels}, codebook {m.rq_codebook_size}",
        f"(+ 4th dedup token), EMA {rq.ema_decay:g}",
        f"item_codes (N+1, {m.rq_levels + 1})",
        f"L ≤ {m.tiger_max_history} items × {m.rq_levels + 1} tokens",
        f"d = {m.d_model}",
        f"{m.tiger_layers} layers × {m.tiger_heads} heads",
        f"beam width W = {m.num_candidates}",
        f"candidate ids (B, K ≤ {m.num_candidates})",
        f"{m.num_time_buckets} log-spaced time buckets",
        f"left padding, L = {m.hstu_max_len}",
        f"HSTU layers × {m.hstu_layers} (d = {m.d_model}, {m.hstu_heads} heads)",
        f"rab_time over {m.num_time_buckets} buckets",
        f"h_user = hidden[last real history token] (B, {m.d_model})",
        f"tabular = [user {USER_FEATURE_DIM} ‖ product {PRODUCT_FEATURE_DIM}]",
        f"(B, K, {fusion})",
        f"{m.ple_task_experts} task expert + {m.ple_shared_experts} shared experts",
        f"expert MLP {fusion}→{m.ple_expert_hidden}→{m.ple_expert_dim}",
        f"Towers ({m.ple_tower_hidden} → 1 / 3)",
        f"z1 ← z1 + log r,  r = {cfg.downsample.rate:g}",
        f"fallback to Platt when weighted positives < {cal.min_positives_isotonic:g}",
        f"H = {ub.horizon_years:g} y, H_hold = {ub.hold_years_mortgage:g} y",
        f"− c_pull {ub.hard_pull_cost:g} (×{ub.fatigue_multiplier:g} if ≥ {ub.fatigue_pulls} pulls)",  # noqa: E501
        f"α = {ut.alpha:g}",
        f"δ = {ut.delta:g} $",
        f"else U −= {ut.harm_penalty:g} $",
        f"NB −= {ut.pending_family_penalty:g} $",
        f"Pre-LN transformer d = {m.prm_d_model}",
        f"{m.prm_layers} layer × {m.prm_heads} heads",
        f"scores (B, {m.slate_size})",
        f"β = {m.prm_cannibalization_weight:g}",
        f"slate order (B, {m.slate_size})",
        f"item_ids ({m.slate_size},)",
    ]
    training = [
        f"snapshot (= {365:g} d)",
        f"K = {m.num_candidates} candidates",
        f"w_floor = {cfg.pending.w_floor:g}",
        f"catalog features (N+1, {PRODUCT_FEATURE_DIM})",
        f"loss = recon MSE + {rq.beta_commit:g} · commitment",
        f"codebooks: EMA {rq.ema_decay:g}",
        f"dead-code reset every {rq.dead_code_reset_steps} steps",
        f"Adam {sci(qo.lr)}, wd {qo.weight_decay:g}, full batch, {thousands(qo.total_steps)} steps",
        f"utilization ≥ {pct(cfg.rqvae_min_utilization)}",
        f"item_codes (N+1, {m.rq_levels + 1}) [{m.rq_levels} levels + dedup]",
        f"each level's {m.rq_codebook_size} tokens",
        f"AdamW {sci(to.lr)}, wd {to.weight_decay:g}",
        f"warmup {to.warmup_steps} → cosine to {pct(to.final_lr_fraction)}, clip {to.grad_clip:.1f}, batch {to.batch_size}",  # noqa: E501
        f"(patience {to.patience})",
        f"r = {cfg.downsample.rate:g}",
        f"λ_click = {fl.lambda_click:g}",
        f"λ_apply = {fl.lambda_apply:g}",
        f"λ_approve = {fl.lambda_approve:g}",
        f"μ_ctcvr = {fl.mu_ctcvr:g}",
        f"μ_ctcavr = {fl.mu_ctcavr:g}",
        f"λ_amount = {fl.lambda_amount:g}",
        f"AdamW {sci(ro.lr)}, wd {ro.weight_decay:g}, warmup {ro.warmup_steps} → cosine {pct(ro.final_lr_fraction)}, "  # noqa: E501
        f"clip {ro.grad_clip:.1f}, batch {ro.batch_size} slates × K = {m.num_candidates}",
        f"calibration split ({pct(cfg.split.calib)} of users)",
        f"fallback < {cal.min_positives_isotonic:g} positives",
        f"softmax(U / {m.prm_utility_temperature:g})",
        f"λ = {prm.loss_cannibalization_weight:g})",
        f"AdamW {sci(po.lr)}, wd {po.weight_decay:g}, warmup {po.warmup_steps} → cosine {pct(po.final_lr_fraction)}, "  # noqa: E501
        f"clip {po.grad_clip:.1f}, batch {po.batch_size} slates",
        f"β = {m.prm_cannibalization_weight:g} rerank is inference-only",
        f"served_at ~ U[snapshot − {90:g}, snapshot]",
    ]
    assert (
        DelayConfig().instant_delay_days > 0
    )  # the delay law is documented in the write-up, not drawn  # noqa: E501
    return {"serving_pipeline.svg": serving, "training_pipeline.svg": training}


def main() -> int:
    problems: list[str] = []
    svgs = {
        name: (DIAGRAMS / name).read_text(encoding="utf-8")
        for name in sorted(hyperparameter_strings())
    }  # noqa: E501
    all_refs: dict[str, set[str]] = {}
    for name, svg in svgs.items():
        size = len(svg.encode("utf-8"))
        if size > MAX_BYTES:
            problems.append(f"{name}: {size} bytes > {MAX_BYTES}")
        if f"width='{WIDTH}'" not in svg:
            problems.append(f"{name}: width is not {WIDTH}")
        for px in re.findall(r"font-size='([\d.]+)px'", svg):
            if float(px) < MIN_FONT_PX:
                problems.append(f"{name}: font-size {px}px < {MIN_FONT_PX}px")
                break
        lines = texts(svg)
        seen_symbol_lines: dict[str, int] = {}
        refs: set[str] = set()
        for body, is_symbol in lines:
            for path, symbol in references(body):
                refs.add(f"{path}::{symbol}")
                err = resolve(path, symbol)
                if err:
                    problems.append(f"{name}: {err}")
            if is_symbol and "::" in body:
                seen_symbol_lines[body] = seen_symbol_lines.get(body, 0) + 1
        for body, n in seen_symbol_lines.items():
            if n > 1:
                problems.append(f"{name}: symbol line drawn {n} times: {body!r}")
        all_refs[name] = refs
        for needle in hyperparameter_strings()[name]:
            if not any(needle in body for body, _ in lines):
                problems.append(
                    f"{name}: hyper-parameter text missing or not equal to config: {needle!r}"
                )  # noqa: E501
        for legend in (
            "solid arrow = tensor flow",
            "dashed arrow = lookup / config / mask",
            "double border = learned module",
            "grey fill = precomputed / frozen at serve time",
            "red outline = compliance gate",
        ):
            if not any(legend in body for body, _ in lines):
                problems.append(f"{name}: legend entry missing: {legend!r}")
    serving_text = " ".join(b for b, _ in texts(svgs["serving_pipeline.svg"]))
    for gate in ("GATE ①", "GATE ②", "GATE ③"):
        if gate not in serving_text:
            problems.append(f"serving_pipeline.svg: missing enforcement point {gate}")
    union = set().union(*all_refs.values())
    referenced_names = {r.split("::")[1].split(".")[0] for r in union}
    for comp, path in required_components().items():
        if comp not in referenced_names:
            problems.append(f"component {path}::{comp} appears in no diagram")
    for p in problems:
        print("FAIL", p)
    if not problems:
        n_comp, n_ref = len(required_components()), len(union)
        print(f"ok: {n_comp} components, {n_ref} references, both diagrams pass")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
