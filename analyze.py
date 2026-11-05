#!/usr/bin/env python3
"""
Aggregate completed runs into the manuscript's tables, statistics and figures.

  python analyze.py --out-root outputs --report report

Reads only runs that finished (test_metrics.json present). Writes to --report:
  table_metrics.csv        seven test metrics, mean +/- 95% CI over seeds   (Table 5, Figure 6)
  table_stats.csv          Welch t-tests, paired t-tests, Cohen's d, Holm-adjusted p (Section 5.5)
  table_epsilon.csv        privacy-budget sensitivity                         (Table 3)
  table_communication.csv  payload and total communication                    (Table 4, Figure 5)
  fig3..fig9 *.png/*.pdf   Figures 3-9 at 300 dpi
  summary.txt              every value quoted in the text, grouped by RQ1-RQ4

All statistics use the seed (independent repetition) as the unit of analysis;
clients are never treated as independent samples.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc_roc", "auprc"]
METRIC_LABEL = {"accuracy": "Accuracy (%)", "sensitivity": "Sensitivity (%)",
                "specificity": "Specificity (%)", "precision": "Precision (%)", "f1": "F1-score (%)",
                "auc_roc": "AUC-ROC", "auprc": "AUPRC"}
LABEL = {"cl": "CL (pooled data, reference)", "fedavg_noniid": "FL FedAvg (non-IID)",
         "fedprox_noniid": "FL FedProx (non-IID)", "fedavg_iid": "FL FedAvg (IID)",
         "fedavg_noniid_dp": "FL FedAvg + DP (σ = 1.1)", "fedavg_noniid_eps0.5": "FL + DP (ε = 0.5)",
         "fedavg_noniid_eps1.0": "FL + DP (ε = 1.0)", "fedavg_noniid_eps2.0": "FL + DP (ε = 2.0)",
         "fedavg_noniid_fp16": "FL FedAvg (FP16)"}
DATASET_LABEL = {"mitbih": "MIT-BIH arrhythmia", "mimic": "MIMIC-III readmission",
                 "isic": "ISIC 2019 skin lesion", "radiology": "Synthetic radiology"}
# fixed colour per entity (validated categorical palette); line style is a second encoding
COLOR = {"fedavg_noniid": "#2a78d6", "fedprox_noniid": "#eb6834", "cl": "#1baf7a",
         "fedavg_iid": "#eda100", "fedavg_noniid_dp": "#6250d6",
         "fedavg_noniid_eps0.5": "#9ec5f4", "fedavg_noniid_eps1.0": "#5598e7",
         "fedavg_noniid_eps2.0": "#1c5cab", "fedavg_noniid_fp16": "#e87ba4"}
STYLE = {"fedavg_noniid": "-", "fedprox_noniid": "--", "cl": ":", "fedavg_iid": "-.",
         "fedavg_noniid_dp": "-", "fedavg_noniid_eps0.5": "--", "fedavg_noniid_eps1.0": "-.",
         "fedavg_noniid_eps2.0": ":"}
COMPARISONS = [("fedavg_noniid", "cl", "RQ1: FL (FedAvg) vs centralized"),
               ("fedprox_noniid", "fedavg_noniid", "RQ1: FedProx vs FedAvg"),
               ("fedavg_noniid_dp", "fedavg_noniid", "RQ2: FL + DP vs FL"),
               ("fedavg_iid", "fedavg_noniid", "RQ3: IID vs non-IID"),
               ("fedavg_noniid_fp16", "fedavg_noniid", "RQ4: FP16 vs FP32")]


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return v[0], np.nan, 1
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)), len(v)


def fmt(m, h, digits=2):
    if np.isnan(m):
        return "–"
    return f"{m:.{digits}f}" + ("" if np.isnan(h) else f" ± {h:.{digits}f}")


def holm(pvals):
    """Holm-Bonferroni adjustment; undefined p-values stay undefined and are not counted."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        adj[i] = running
    return adj


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


# ------------------------------------------------------------------ loading
def load(root):
    tests, rounds, clients, comms, parts = [], [], [], [], []
    for tm in glob.glob(os.path.join(root, "*", "*", "seed*", "test_metrics.json")):
        run = os.path.dirname(tm)
        ds, cond = run.split(os.sep)[-3], run.split(os.sep)[-2]
        seed = int(os.path.basename(run)[4:])
        t = json.load(open(tm)); t.update(dataset=ds, condition=cond, seed=seed); tests.append(t)
        r = pd.read_csv(os.path.join(run, "round_log.csv")); r["dataset"], r["condition"], r["seed"] = ds, cond, seed
        rounds.append(r)
        c = json.load(open(os.path.join(run, "comm.json"))); c.update(dataset=ds, condition=cond, seed=seed)
        comms.append(c)
        cm = os.path.join(run, "client_metrics.csv")
        if os.path.exists(cm):
            x = pd.read_csv(cm); x["dataset"], x["condition"], x["seed"] = ds, cond, seed; clients.append(x)
        x = pd.read_csv(os.path.join(run, "partition.csv")); x["dataset"], x["condition"], x["seed"] = ds, cond, seed
        parts.append(x)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return (pd.DataFrame(tests), cat(rounds), cat(clients), pd.DataFrame(comms), cat(parts))


def with_all_datasets(tests):
    """Add an 'all' pseudo-dataset: per seed, the mean over datasets (only where every dataset ran)."""
    n_ds = tests["dataset"].nunique()
    g = tests.groupby(["condition", "seed"])
    full = g.filter(lambda x: x["dataset"].nunique() == n_ds)
    if n_ds < 2 or full.empty:
        return tests
    allrow = full.groupby(["condition", "seed"])[METRICS].mean().reset_index()
    allrow["dataset"] = "all"
    return pd.concat([tests, allrow], ignore_index=True)


# ------------------------------------------------------------------ tables
def table_metrics(tests):
    rows = []
    for (ds, cond), g in tests.groupby(["dataset", "condition"]):
        row = {"dataset": ds, "condition": cond, "n_seeds": g["seed"].nunique()}
        for m in METRICS:
            mu, h, _ = mean_ci(g[m])
            row[m] = fmt(mu, h, 2 if m in ("auc_roc", "auprc") else 1) if m not in ("auc_roc", "auprc") \
                else fmt(mu, h, 3)
            row[m + "_mean"], row[m + "_ci95"] = mu, h
        rows.append(row)
    return pd.DataFrame(rows)


def table_stats(tests):
    rows = []
    for ds in sorted(tests["dataset"].unique()):
        d = tests[tests["dataset"] == ds]
        for a_c, b_c, name in COMPARISONS:
            A, B = d[d.condition == a_c].set_index("seed"), d[d.condition == b_c].set_index("seed")
            if len(A) < 2 or len(B) < 2:
                continue
            block = []
            for m in METRICS:
                a, b = A[m].dropna().values, B[m].dropna().values
                welch = stats.ttest_ind(a, b, equal_var=False)
                common = A.index.intersection(B.index)
                paired = (stats.ttest_rel(A.loc[common, m], B.loc[common, m]).pvalue
                          if len(common) >= 2 else np.nan)
                diff = a.mean() - b.mean()
                se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                dof = welch.df if hasattr(welch, "df") else len(a) + len(b) - 2
                block.append({"dataset": ds, "comparison": name, "a": a_c, "b": b_c, "metric": m,
                              "mean_a": a.mean(), "mean_b": b.mean(), "difference": diff,
                              "diff_ci95_low": diff - stats.t.ppf(0.975, dof) * se,
                              "diff_ci95_high": diff + stats.t.ppf(0.975, dof) * se,
                              "welch_t": welch.statistic, "p_welch": welch.pvalue,
                              "p_paired": paired, "cohens_d": cohens_d(a, b),
                              "n_a": len(a), "n_b": len(b)})
            adj = holm([r["p_welch"] for r in block])
            for r, p in zip(block, adj):
                r["p_welch_holm"] = p
            rows += block
    return pd.DataFrame(rows)


def table_epsilon(tests):
    rows = []
    order = ["fedavg_noniid_eps0.5", "fedavg_noniid_eps1.0", "fedavg_noniid_eps2.0",
             "fedavg_noniid_dp", "fedavg_noniid"]
    for ds in sorted(tests["dataset"].unique()):
        d = tests[tests.dataset == ds]
        base = d[d.condition == "fedavg_noniid"]["accuracy"].mean()
        for cond in order:
            g = d[d.condition == cond]
            if g.empty:
                continue
            acc, acc_h, _ = mean_ci(g["accuracy"]); pr, pr_h, _ = mean_ci(g["auprc"])
            rows.append({"dataset": ds, "condition": cond,
                         "sigma_mean": g["sigma_mean"].mean() if "sigma_mean" in g and g["sigma_mean"].notna().any() else np.nan,
                         "epsilon_achieved": g["epsilon"].mean() if g["epsilon"].notna().any() else np.nan,
                         "accuracy": fmt(acc, acc_h, 1), "auprc": fmt(pr, pr_h, 3),
                         "accuracy_change_vs_no_dp_pp": acc - base if cond != "fedavg_noniid" else 0.0})
    return pd.DataFrame(rows)


def table_comm(comms, tests):
    rows = []
    for ds in sorted(comms["dataset"].unique()):
        c = comms[comms.dataset == ds]
        fp32 = c[c.condition == "fedavg_noniid"]
        fp16 = c[c.condition == "fedavg_noniid_fp16"]
        cl = c[c.condition == "cl"]
        if fp32.empty:
            continue
        f = fp32.iloc[0]
        row = {"dataset": ds, "n_params": int(f.n_params),
               "payload_per_client_round_MB_fp32": f.payload_fp32_bytes / 1e6,
               "payload_per_client_round_MB_fp16": (fp16.iloc[0].payload_tx_bytes_per_direction / 1e6
                                                    if not fp16.empty else np.nan),
               "fl_total_GB_fp32": f.fl_total_bytes / 1e9,
               "fl_total_GB_fp16": fp16.iloc[0].fl_total_bytes / 1e9 if not fp16.empty else np.nan,
               "cl_raw_data_GB": (cl.iloc[0].cl_raw_data_bytes if not cl.empty else f.cl_raw_data_bytes) / 1e9}
        row["fl_to_cl_ratio"] = row["fl_total_GB_fp32"] / row["cl_raw_data_GB"] if row["cl_raw_data_GB"] else np.nan
        row["reduction_vs_cl_percent"] = 100 * (1 - row["fl_to_cl_ratio"])
        t = tests[tests.dataset == ds]
        a32, a16 = t[t.condition == "fedavg_noniid"].accuracy, t[t.condition == "fedavg_noniid_fp16"].accuracy
        row["fp16_accuracy_change_pp"] = a16.mean() - a32.mean() if len(a16) and len(a32) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
from matplotlib.ticker import MaxNLocator  # noqa: E402


def _style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35,
                         "grid.linestyle": "--", "lines.linewidth": 2, "savefig.dpi": 300,
                         "legend.frameon": False})


def _save(fig, report, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(report, f"{name}.{ext}"), dpi=300)


def _curve(ax, rounds, ds, cond, col="val_acc", label=None):
    g = rounds[(rounds.dataset == ds) & (rounds.condition == cond)]
    if g.empty:
        return None
    rows = [(r, *mean_ci(x[col])[:2]) for r, x in g.groupby("round")]
    arr = np.array(rows, dtype=float)
    final_m, final_h = arr[-1, 1], arr[-1, 2]
    ax.plot(arr[:, 0], arr[:, 1], color=COLOR.get(cond, "#52514e"), ls=STYLE.get(cond, "-"),
            label=(label or LABEL[cond]) + (f": {final_m:.1f}%" + ("" if np.isnan(final_h) else f" ± {final_h:.1f}")
                                            if col == "val_acc" else ""))
    if not np.all(np.isnan(arr[:, 2])):
        lo, hi = arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2]
        if col == "val_acc":                      # accuracy is bounded; clip the band to [0, 100]
            lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
        ax.fill_between(arr[:, 0], lo, hi, color=COLOR.get(cond, "#52514e"), alpha=0.18, lw=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return arr


def figures(tests, rounds, clients, comms, parts, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    made = []
    for ds in sorted(tests["dataset"].unique()):
        if ds == "all":
            continue
        name = DATASET_LABEL.get(ds, ds)
        # Figure 3 - accuracy across rounds (FedAvg, FedProx, CL)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if any(_curve(ax, rounds, ds, c) is not None for c in ("fedavg_noniid", "fedprox_noniid", "cl")):
            ax.set(xlabel="Communication round (CL: epoch)", ylabel="Validation accuracy (%)", title=name)
            ax.legend(loc="lower right", fontsize=8.5, title="Final value at last round", title_fontsize=8)
            _save(fig, report, f"fig3_accuracy_rounds_{ds}"); made.append(f"fig3_accuracy_rounds_{ds}")
        plt.close(fig)

        # Figure 4 - privacy budget across rounds (DP runs only; non-DP runs have no finite epsilon)
        fig, ax = plt.subplots(figsize=(7, 4.2)); any_dp = False
        for cond in ("fedavg_noniid_dp", "fedavg_noniid_eps0.5", "fedavg_noniid_eps1.0", "fedavg_noniid_eps2.0"):
            g = rounds[(rounds.dataset == ds) & (rounds.condition == cond)]
            if g.empty or g["epsilon"].isna().all():
                continue
            any_dp = True
            m = g.groupby("round")["epsilon"].mean()
            ax.plot(m.index, m.values, color=COLOR[cond], ls=STYLE[cond],
                    label=f"{LABEL[cond]}: final ε = {m.values[-1]:.2f}")
        if any_dp:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
            ax.set(xlabel="Communication round", ylabel="Cumulative privacy budget ε (δ = 10⁻⁵)", title=name)
            ax.legend(fontsize=8.5)
            _save(fig, report, f"fig4_epsilon_rounds_{ds}"); made.append(f"fig4_epsilon_rounds_{ds}")
        plt.close(fig)

        # Figure 6 - classification metrics (two panels: % metrics and AUC metrics)
        conds = [c for c in ("fedavg_noniid", "fedprox_noniid", "cl") if c in set(tests[tests.dataset == ds].condition)]
        if conds:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [5, 2]})
            for ax, ms, scale in ((a1, METRICS[:5], "%"), (a2, METRICS[5:], "")):
                x = np.arange(len(ms)); w = 0.8 / len(conds)
                for i, c in enumerate(conds):
                    g = tests[(tests.dataset == ds) & (tests.condition == c)]
                    mc = [mean_ci(g[m])[:2] for m in ms]
                    top = 100.0 if scale else 1.0
                    means = np.array([v[0] for v in mc]); half = np.nan_to_num([v[1] for v in mc])
                    yerr = [np.minimum(half, means), np.minimum(half, top - means)]   # keep bars inside [0, top]
                    ax.bar(x + (i - (len(conds) - 1) / 2) * w, means, w * 0.92, yerr=yerr, capsize=2,
                           color=COLOR[c], label=LABEL[c], edgecolor="white", linewidth=0.8,
                           error_kw=dict(ecolor="#52514e", lw=1))
                ax.set_ylim(0, 100 if scale else 1.0)
                ax.set_xticks(x); ax.set_xticklabels([METRIC_LABEL[m].replace(" (%)", "") for m in ms], fontsize=8.5)
                ax.grid(axis="x", visible=False)
                ax.set_ylabel("Score (%)" if scale else "Score")
            a1.legend(fontsize=8, loc="lower left"); fig.suptitle(name, fontsize=10)
            _save(fig, report, f"fig6_metrics_{ds}"); made.append(f"fig6_metrics_{ds}")
            plt.close(fig)

        # Figure 7 - IID vs non-IID convergence, gap computed from final-round values per seed
        fig, ax = plt.subplots(figsize=(7, 4.3))
        iid = _curve(ax, rounds, ds, "fedavg_iid"); non = _curve(ax, rounds, ds, "fedavg_noniid")
        _curve(ax, rounds, ds, "cl")
        if iid is not None and non is not None:
            last = int(min(iid[-1, 0], non[-1, 0]))
            g = rounds[(rounds.dataset == ds) & (rounds["round"] == last)]
            a = g[g.condition == "fedavg_iid"]["val_acc"].values
            b = g[g.condition == "fedavg_noniid"]["val_acc"].values
            ia, nb = a.mean(), b.mean()
            ax.annotate("", xy=(last, ia), xytext=(last, nb), arrowprops=dict(arrowstyle="<->", lw=1.2))
            txt = f"{ia - nb:.1f} pp"
            if len(a) > 1 and len(b) > 1:
                p = stats.ttest_ind(a, b, equal_var=False).pvalue
                txt += f"\np = {p:.3f}" if p >= 0.001 else "\np < 0.001"
                txt += f"\nd = {cohens_d(a, b):.2f}"
            ax.text(last + 0.8, (ia + nb) / 2, txt, ha="left", va="center", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#b0afa9"))
            ax.set_xlim(1, last * 1.16 + 1)
            ax.set(xlabel="Communication round (CL: epoch)", ylabel="Validation accuracy (%)", title=name)
            ax.legend(loc="lower right", fontsize=8.5, title="Final value at last round", title_fontsize=8)
            _save(fig, report, f"fig7_iid_vs_noniid_{ds}"); made.append(f"fig7_iid_vs_noniid_{ds}")
        plt.close(fig)

        # Figure 8 - client class distribution (first available seed of the non-IID run)
        p = parts[(parts.dataset == ds) & (parts.condition == "fedavg_noniid")]
        if not p.empty:
            p = p[p.seed == p.seed.min()]
            cls = [c for c in p.columns if c.startswith("class_")]
            mat = p[cls].values / p[cls].values.sum(1, keepdims=True)
            fig, ax = plt.subplots(figsize=(1.2 + 0.7 * len(cls), 4.2))
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(cls))); ax.set_xticklabels([c.replace("class_", "") for c in cls])
            ax.set_yticks(range(len(p))); ax.set_yticklabels([f"Client {k}" for k in p.client])
            ax.set(xlabel="Class", title=f"{name}: class share per client (seed {p.seed.min()})")
            ax.grid(False)
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "#0b0b0b")
            fig.colorbar(im, ax=ax, fraction=0.04, label="Proportion of client's samples")
            _save(fig, report, f"fig8_client_classes_{ds}"); made.append(f"fig8_client_classes_{ds}")
            plt.close(fig)

        # Figure 9 - per-client accuracy of the final global model (clients x seeds)
        c = clients[(clients.dataset == ds) & (clients.condition == "fedavg_noniid")] if not clients.empty else clients
        if not c.empty:
            data = [g["val_acc"].values for _, g in c.groupby("client")]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="#cde2fb", color="#2a78d6"),
                       medianprops=dict(color="#0d366b", lw=2), whiskerprops=dict(color="#2a78d6"),
                       capprops=dict(color="#2a78d6"), flierprops=dict(markeredgecolor="#2a78d6"))
            ax.set_xticklabels([f"C{k}" for k in sorted(c.client.unique())])
            allv = c["val_acc"].values
            ax.set(xlabel="Client", ylabel="Local validation accuracy (%)",
                   title=f"{name}: range {allv.min():.1f}–{allv.max():.1f}%, median {np.median(allv):.1f}% "
                         f"({c.client.nunique()} clients × {c.seed.nunique()} seeds)")
            ax.title.set_fontsize(9)
            _save(fig, report, f"fig9_client_accuracy_{ds}"); made.append(f"fig9_client_accuracy_{ds}")
            plt.close(fig)

    # Figure 5 - communication (small multiples: each dataset has its own scale)
    tc = table_comm(comms, tests)
    if not tc.empty:
        fig, axes = plt.subplots(1, len(tc), figsize=(2.6 * len(tc) + 0.6, 3.8), squeeze=False)
        for ax, (_, r) in zip(axes[0], tc.iterrows()):
            vals = [("FL FP32", r.fl_total_GB_fp32, "#2a78d6"), ("FL FP16", r.fl_total_GB_fp16, "#e87ba4"),
                    ("CL raw data", r.cl_raw_data_GB, "#1baf7a")]
            vals = [v for v in vals if not np.isnan(v[1])]
            ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals],
                   edgecolor="white", linewidth=0.8)
            for i, v in enumerate(vals):
                ax.text(i, v[1], f"{v[1]:.3g}", ha="center", va="bottom", fontsize=8)
            ax.set_title(DATASET_LABEL.get(r.dataset, r.dataset), fontsize=9)
            ax.set_ylabel("Data transferred (GB)"); ax.tick_params(axis="x", labelsize=8, rotation=20)
            ax.grid(axis="x", visible=False)
        _save(fig, report, "fig5_communication"); made.append("fig5_communication")
        plt.close(fig)
    return made


# ------------------------------------------------------------------ summary
def summary(tests, st, te, tc, clients):
    L = ["Values computed from the logged runs. Copy them into the manuscript; do not edit them by hand.", ""]
    for ds in sorted(tests["dataset"].unique()):
        d = tests[tests.dataset == ds]
        L.append(f"=== {DATASET_LABEL.get(ds, 'All datasets (mean over datasets per seed)')} ===")
        for cond in ("cl", "fedavg_noniid", "fedprox_noniid", "fedavg_iid", "fedavg_noniid_dp"):
            g = d[d.condition == cond]
            if g.empty:
                continue
            parts = [f"{METRIC_LABEL[m].split(' (')[0]} {fmt(*mean_ci(g[m])[:2], 3 if m in ('auc_roc', 'auprc') else 1)}"
                     for m in METRICS]
            L.append(f"  {LABEL[cond]} (n = {g.seed.nunique()} seeds): " + "; ".join(parts))
        if not st.empty:
            for _, r in st[(st.dataset == ds) & (st.metric == "accuracy")].iterrows():
                L.append(f"  {r.comparison}: accuracy difference {r.difference:+.2f} pp "
                         f"(95% CI {r.diff_ci95_low:+.2f} to {r.diff_ci95_high:+.2f}); Welch p = {r.p_welch:.3f}, "
                         f"Holm-adjusted p = {r.p_welch_holm:.3f}, paired p = {r.p_paired:.3f}; Cohen's d = {r.cohens_d:.2f}")
        if not te.empty and ds in set(te.dataset):
            for _, r in te[te.dataset == ds].iterrows():
                L.append(f"  RQ2 {LABEL[r.condition]}: σ = {r.sigma_mean:.2f}, achieved ε = {r.epsilon_achieved:.2f}, "
                         f"accuracy {r.accuracy}, AUPRC {r.auprc}, change vs no DP {r.accuracy_change_vs_no_dp_pp:+.2f} pp"
                         if not np.isnan(r.epsilon_achieved) else
                         f"  RQ2 no-DP baseline: accuracy {r.accuracy}, AUPRC {r.auprc}")
        if not tc.empty and ds in set(tc.dataset):
            r = tc[tc.dataset == ds].iloc[0]
            nf = lambda v, f: "not run" if pd.isna(v) else format(v, f)  # noqa: E731
            L.append(f"  RQ4 payload per client per round: {nf(r.payload_per_client_round_MB_fp32, '.2f')} MB (FP32), "
                     f"{nf(r.payload_per_client_round_MB_fp16, '.2f')} MB (FP16); FL total "
                     f"{nf(r.fl_total_GB_fp32, '.3f')} GB vs CL raw data {nf(r.cl_raw_data_GB, '.3f')} GB "
                     f"(FL/CL = {nf(r.fl_to_cl_ratio, '.2f')}); FP16 accuracy change "
                     f"{nf(r.fp16_accuracy_change_pp, '+.2f')} pp")
        if not clients.empty and ds in set(clients.dataset):
            c = clients[(clients.dataset == ds) & (clients.condition == "fedavg_noniid")]["val_acc"]
            if len(c):
                L.append(f"  RQ3 per-client accuracy (FedAvg non-IID): {c.min():.1f}–{c.max():.1f}%, median {c.median():.1f}%")
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--report", default="report")
    a = p.parse_args()
    os.makedirs(a.report, exist_ok=True)
    tests, rounds, clients, comms, parts = load(a.out_root)
    if tests.empty:
        raise SystemExit(f"No completed runs found under {a.out_root}/")
    tests_all = with_all_datasets(tests)
    tm, st, te, tc = table_metrics(tests_all), table_stats(tests_all), table_epsilon(tests), table_comm(comms, tests)
    tm.to_csv(os.path.join(a.report, "table_metrics.csv"), index=False)
    st.to_csv(os.path.join(a.report, "table_stats.csv"), index=False)
    te.to_csv(os.path.join(a.report, "table_epsilon.csv"), index=False)
    tc.to_csv(os.path.join(a.report, "table_communication.csv"), index=False)
    made = figures(tests, rounds, clients, comms, parts, a.report)
    s = summary(tests_all, st, te, tc, clients)
    open(os.path.join(a.report, "summary.txt"), "w").write(s)
    print(s)
    print(f"Runs analysed: {len(tests)}; figures: {len(made)}; report written to {a.report}/")


if __name__ == "__main__":
    main()
