#!/usr/bin/env python3
"""
Run the full experiment grid of the manuscript (Sections 5-6).

For every dataset x seed the following conditions are run:
  cl                     centralized reference upper bound            (RQ1)
  fedavg_noniid          FedAvg, Dirichlet alpha = 0.5                 (RQ1, primary)
  fedprox_noniid         FedProx mu = 0.01, Dirichlet alpha = 0.5      (RQ1)
  fedavg_iid             FedAvg, IID partition (ablation)              (RQ3)
  fedavg_noniid_dp       FedAvg + DP-SGD, sigma = 1.1, C = 1.0         (RQ2)
  fedavg_noniid_eps0.5   FedAvg + DP-SGD calibrated to epsilon = 0.5   (RQ2, Table 3)
  fedavg_noniid_eps1.0   ... epsilon = 1.0
  fedavg_noniid_eps2.0   ... epsilon = 2.0
  fedavg_noniid_fp16     FedAvg with FP16 model transmission           (RQ4, Table 4)

Finished runs (test_metrics.json present) are skipped, so the script can be
interrupted and restarted. Extra arguments after "--" go to run_experiment.py.

  python run_all.py --datasets mitbih mimic --seeds 42 123 456 789 1024 -- --client-gpus 0.2
  python run_all.py --dry-run
"""
import argparse
import os
import subprocess
import sys

CONDITIONS = {
    "cl":                   ["--mode", "cl"],
    "fedavg_noniid":        ["--mode", "fedavg", "--partition", "dirichlet"],
    "fedprox_noniid":       ["--mode", "fedprox", "--partition", "dirichlet"],
    "fedavg_iid":           ["--mode", "fedavg", "--partition", "iid"],
    "fedavg_noniid_dp":     ["--mode", "fedavg", "--partition", "dirichlet", "--dp"],
    "fedavg_noniid_eps0.5": ["--mode", "fedavg", "--partition", "dirichlet", "--target-epsilon", "0.5"],
    "fedavg_noniid_eps1.0": ["--mode", "fedavg", "--partition", "dirichlet", "--target-epsilon", "1.0"],
    "fedavg_noniid_eps2.0": ["--mode", "fedavg", "--partition", "dirichlet", "--target-epsilon", "2.0"],
    "fedavg_noniid_fp16":   ["--mode", "fedavg", "--partition", "dirichlet", "--fp16"],
}
DATASETS = ["mitbih", "mimic", "isic", "radiology"]
SEEDS = [42, 123, 456, 789, 1024]


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--"); argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    jobs = [(d, c, s) for d in a.datasets for s in a.seeds for c in a.conditions]
    for n, (d, c, s) in enumerate(jobs, 1):
        out = os.path.join(a.out_root, d, c, f"seed{s}")
        if os.path.exists(os.path.join(out, "test_metrics.json")):
            print(f"[{n}/{len(jobs)}] skip {out} (done)")
            continue
        cmd = [sys.executable, os.path.join(here, "run_experiment.py"), "--dataset", d,
               "--seed", str(s), "--out", out, "--data-dir", a.data_dir, *CONDITIONS[c], *extra]
        print(f"[{n}/{len(jobs)}] {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "stdout.txt"), "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"    FAILED (exit {rc}) - see {out}/stdout.txt", flush=True)


if __name__ == "__main__":
    main()
