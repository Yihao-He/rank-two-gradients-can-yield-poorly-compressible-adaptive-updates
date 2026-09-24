"""Recompute analytic residuals and archived-gradient precision diagnostics."""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MATRICES = RESULTS / "gradient_matrices"

def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    curve = []
    for n in (32, 64, 128, 256, 512):
        s = 1 / np.sin((2*np.arange(1, n+1)-1)*np.pi/(2*n))
        s = np.sort(s)[::-1]
        for r in sorted(set([1,2,4,8,16,32,64,128,256,n])):
            if r <= n:
                curve.append({"n": n, "rank": r,
                    "best_rank_relative_frobenius_residual": np.sqrt(np.square(s[r:]).sum()/np.square(s).sum())})
    write_rows(RESULTS / "analytic_residual_curve.csv", curve)

    audits = []
    for block in ("last_layer_down_proj", "layer_minus_4_down_proj"):
        with np.load(MATRICES / f"{block}_gradient_history.npz") as z:
            history = z["gradients"].astype(np.float64)
        with np.load(MATRICES / f"{block}_final_gradient_diagnostics.npz") as z:
            A = z["gradient"].astype(np.float64)
            u, s, vh = z["u"].astype(np.float64), z["singular_values"].astype(np.float64), z["vh"].astype(np.float64)
            saved_rank2 = z["rank2_reconstruction"].astype(np.float64)
        assert np.array_equal(history[-1], A)
        _, scpu, _ = np.linalg.svd(A, full_matrices=False)
        norm = np.linalg.norm(A)
        gpu_rank2 = (u[:, :2] * s[:2]) @ vh[:2]
        audits.append({
            "block": block, "shape": list(A.shape), "dtype": "archived float32; evaluated in float64",
            "cpu_best_rank_residuals": {str(r): float(np.linalg.norm(scpu[r:])/norm) for r in (2,4,8,16,32)},
            "gpu_singular_value_rank2_tail": float(np.linalg.norm(s[2:])/np.linalg.norm(s)),
            "gpu_saved_rank2_reconstruction_residual": float(np.linalg.norm(A-saved_rank2)/norm),
            "gpu_factor_product_recomputed_in_float64_residual": float(np.linalg.norm(A-gpu_rank2)/norm),
            "gpu_rank2_U_orthogonality_error": float(np.linalg.norm(u[:,:2].T@u[:,:2]-np.eye(2))),
            "gpu_rank2_V_orthogonality_error": float(np.linalg.norm(vh[:2]@vh[:2].T-np.eye(2))),
        })
    (RESULTS / "gradient_precision_audit.json").write_text(json.dumps({
        "method": "CPU float64 SVD and Frobenius tails of the archived float32 matrices.",
        "results": audits}, indent=2) + "\n", encoding="utf-8")
    rows = []
    for item in audits:
        for rank in (2,4,8,16,32):
            rows.append({"block": item["block"], "rank": rank,
                         "cpu_float64_best_rank_residual": item["cpu_best_rank_residuals"][str(rank)]})
    write_rows(RESULTS / "gradient_precision_comparison.csv", rows)

if __name__ == "__main__":
    main()
