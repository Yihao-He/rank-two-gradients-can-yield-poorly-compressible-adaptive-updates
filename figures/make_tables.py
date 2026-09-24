"""Generate the damping and optimizer-stage tables from retained measurements."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def number(x):
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        power = int(np.floor(np.log10(abs(x))))
        return "$" + rf"{x / 10**power:.2f}\times10^{{{power}}}$"
    return f"{x:.4f}"


def main():
    cpu = json.loads((OUT / "source_cpu_summary.json").read_text(encoding="utf-8"))
    damping = [r for r in cpu["damping_checks_n256"] if r["gradient_scale"] == 1.0]
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"$\epsilon$ & $\|U_\epsilon-U_0\|_F/n$ & Bound $\delta$ & $E_8(U_\epsilon)$\\",
             r"\midrule"]
    for r in damping:
        values = [r["adam_epsilon"], r["relative_perturbation_from_sign"],
                  r["entrywise_perturbation_bound"], r["best_rank_8_update_residual"]]
        lines.append(" & ".join(number(x) for x in values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_damping.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows, controls = [], []
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for block in ("last_layer_down_proj", "layer_minus_4_down_proj"):
        with np.load(OUT / "gradient_matrices" / f"{block}_gradient_history.npz") as f:
            history = f["gradients"].astype(np.float64)
        with np.load(OUT / "gradient_matrices" / f"{block}_adam_update.npz") as f:
            saved_update = f["update"].astype(np.float64)
        m, v = np.zeros_like(history[0]), np.zeros_like(history[0])
        for gradient in history:
            m = beta1*m + (1-beta1)*gradient
            v = beta2*v + (1-beta2)*gradient**2
        mhat, vhat = m/(1-beta1**len(history)), v/(1-beta2**len(history))
        u, s, vh = np.linalg.svd(history[-1], full_matrices=False)
        rank2_gradient = (u[:, :2]*s[:2])@vh[:2]
        reset = rank2_gradient/(np.abs(rank2_gradient)+eps)
        raw_reset = history[-1]/(np.abs(history[-1])+eps)
        controls.append({"block": block,
            "rank2_gradient_relative_residual": float(np.linalg.norm(history[-1]-rank2_gradient)/np.linalg.norm(history[-1])),
            "rank2_vs_raw_zero_history_update_relative_difference": float(np.linalg.norm(reset-raw_reset)/np.linalg.norm(raw_reset))})
        matrices = {
            "G32": history[-1],
            "M32": mhat,
            "rank2_zero_history_update": reset,
            "U32": saved_update,
        }
        for name, matrix in matrices.items():
            s = np.linalg.svd(matrix, compute_uv=False)
            row = {"block": block, "matrix": name,
                   "numerical_rank_relative_1e-5": int(np.sum(s > s[0]*1e-5)),
                   "frobenius_norm": float(np.linalg.norm(s))}
            row.update({f"E{r}": float(np.linalg.norm(s[r:])/np.linalg.norm(s))
                        for r in (2, 4, 8, 16, 32)})
            rows.append(row)
    with (OUT / "optimizer_stage_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = {
        "method": "CPU float64 SVD of archived float32 G32 and U32; M32 computed in float64 from archived histories; reset control normalizes the CPU best rank-two reconstruction of G32 with zero moments and epsilon=1e-8.",
        "history": "32 fixed-checkpoint minibatches per block; beta1=.9, beta2=.999, epsilon=1e-8; M32 includes bias correction.",
        "numerical_rank_definition": "Count singular values strictly greater than 1e-5 times the largest.",
        "rows": rows, "rank2_control_checks": controls,
    }
    (OUT / "optimizer_stage_diagnostics.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    labels = {"G32": r"$G_{32}$", "M32": r"$\widehat M_{32}$",
              "rank2_zero_history_update": r"$Z_2$", "U32": r"$U_{32}$"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Matrix & Num. rank & $E_2$ & $E_8$ & $E_{32}$\\", r"\midrule"]
    for block, title in [("last_layer_down_proj", "Final layer"),
                         ("layer_minus_4_down_proj", "Layer $-4$")]:
        lines.append(r"\multicolumn{5}{l}{\emph{" + title + r"}}\\")
        for r in rows:
            if r["block"] == block:
                lines.append(" & ".join([labels[r["matrix"]],
                    str(r["numerical_rank_relative_1e-5"])] +
                    [number(r[f"E{k}"]) for k in (2, 8, 32)]) + r"\\")
        if block == "last_layer_down_proj":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_optimizer_stages.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
