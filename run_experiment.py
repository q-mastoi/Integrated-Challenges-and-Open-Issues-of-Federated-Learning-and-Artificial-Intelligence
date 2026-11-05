#!/usr/bin/env python3
"""
Run one experiment of the use-case methodology (Section 5).

Modes
  cl        centralized learning on the pooled training data (reference upper bound)
  fedavg    Flower simulation with FedAvg (McMahan et al., 2017)
  fedprox   Flower simulation with FedProx, proximal coefficient mu (Li et al., 2020)

Defaults reproduce Section 5: 10 clients, 50 rounds, 5 local epochs, Adam,
lr = 1e-3, batch 32, dropout 0.3, Dirichlet alpha = 0.5, mu = 0.01,
DP-SGD with sigma = 1.1, C = 1.0, delta = 1e-5.

Outputs (in --out):
  config.json         full configuration and environment versions
  round_log.csv       per round: pooled validation accuracy/loss, epsilon, cumulative bytes
  test_metrics.json   seven metrics of the final global model on the held-out test set
  client_metrics.csv  final global model evaluated on each client's local hold-out (Figure 9)
  partition.csv       per-client sample and class counts (Figure 8)
  comm.json           payload per client per round and total communication (Table 4)

Example
  python run_experiment.py --dataset mitbih --mode fedavg --partition dirichlet \
      --seed 42 --out outputs/mitbih/fedavg_noniid/seed42
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flhc.data import load_task                                    # noqa: E402
from flhc.models import (fp16_roundtrip, get_weights, make_model,  # noqa: E402
                         payload_bytes, set_weights)
from flhc.partition import partition_indices, split_local           # noqa: E402
from flhc.privacy import epsilon_spent, sigma_for_target            # noqa: E402
from flhc.training import (binary_metrics, class_weights, local_train,  # noqa: E402
                           make_loader, predict, steps_per_epoch)

warnings.filterwarnings("ignore", category=UserWarning, module="opacus")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["mitbih", "mimic", "isic", "radiology"])
    p.add_argument("--mode", required=True, choices=["cl", "fedavg", "fedprox"])
    p.add_argument("--partition", default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--data-dir", default="data")
    # federated setup
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--min-client-size", type=int, default=20)
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--cl-epochs", type=int, default=None,
                   help="centralized epochs (default = --rounds, one epoch per logged step)")
    # differential privacy
    p.add_argument("--dp", action="store_true")
    p.add_argument("--noise-multiplier", type=float, default=1.1)
    p.add_argument("--target-epsilon", type=float, default=None,
                   help="calibrate sigma so the final epsilon equals this value")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    # communication
    p.add_argument("--fp16", action="store_true", help="simulate FP16 model transmission")
    # data / model
    p.add_argument("--val-frac", type=float, default=0.1, help="pooled server validation split")
    p.add_argument("--client-val-frac", type=float, default=0.1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--weights", default=None, help="local ResNet-18 ImageNet checkpoint")
    p.add_argument("--no-bandpass", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: subsample the dataset")
    # resources
    p.add_argument("--device", default="auto")
    p.add_argument("--client-cpus", type=float, default=1.0)
    p.add_argument("--client-gpus", type=float, default=0.0)
    p.add_argument("--ray-cpus", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=256)
    a = p.parse_args(argv)
    if a.mode == "fedprox" and a.dp:
        p.error("FedProx + DP-SGD is not supported (see flhc/training.py).")
    if a.target_epsilon is not None:
        a.dp = True
    return a


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def versions():
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda if torch.cuda.is_available() else None,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu": platform.processor() or platform.machine()}
    for mod in ("flwr", "opacus", "torchvision", "numpy", "pandas", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    t0 = time.time()

    task = load_task(a.dataset, a.data_dir, img_size=a.img_size, limit=a.limit,
                     seed=a.seed, bandpass=not a.no_bandpass)

    # pooled server-side validation split (same for every condition of a seed)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(task.y_train))
    n_val = int(round(a.val_frac * len(perm)))
    val_idx, pool_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

    # clients
    if a.mode == "cl":
        clients = [{"train": pool_idx, "val": np.array([], dtype=int)}]
    else:
        parts = partition_indices(task.strat_train[pool_idx], a.clients, a.partition,
                                  a.alpha, seed=a.seed, min_size=a.min_client_size)
        clients = []
        for k, rel in enumerate(parts):
            tr, va = split_local(pool_idx[rel], a.client_val_frac, seed=a.seed * 100 + k)
            clients.append({"train": tr, "val": va})

    with open(os.path.join(a.out, "partition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        fine = sorted(np.unique(task.strat_train))
        w.writerow(["client", "n_train", "n_val", "positive_rate"] + [f"class_{c}" for c in fine])
        for k, c in enumerate(clients):
            lab = task.strat_train[c["train"]]
            w.writerow([k, len(c["train"]), len(c["val"]),
                        round(float(task.y_train[c["train"]].mean()), 4)]
                       + [int((lab == cc).sum()) for cc in fine])

    # model
    global_model, weights_source = make_model(task, a, pretrained=True)
    init_weights = get_weights(global_model)
    p_fp32, p_tx = payload_bytes(init_weights), payload_bytes(init_weights, fp16=a.fp16)
    n_params = int(sum(p.numel() for p in global_model.parameters()))

    # differential privacy calibration
    E = a.local_epochs if a.mode != "cl" else 1
    R = a.rounds if a.mode != "cl" else (a.cl_epochs or a.rounds)
    dp_cfg = []
    for c in clients:
        spe = steps_per_epoch(len(c["train"]), a.batch_size)
        q, total = 1.0 / spe, R * E * spe
        sigma = None
        if a.dp:
            sigma = (sigma_for_target(a.target_epsilon, a.delta, q, total)
                     if a.target_epsilon is not None else a.noise_multiplier)
        dp_cfg.append({"q": q, "spe": spe, "sigma": sigma})

    val_loader = make_loader(task.train, val_idx, a.eval_batch_size, False)
    log_path = os.path.join(a.out, "round_log.csv")
    with open(log_path, "w") as f:
        f.write("round,val_acc,val_loss,val_auc,epsilon,bytes_cum\n")

    def log_round(r, model):
        y, p, loss = predict(model, val_loader, device)
        m = binary_metrics(y, p)
        eps = (max(epsilon_spent(d["sigma"], d["q"], r * E * d["spe"], a.delta) for d in dp_cfg)
               if a.dp else "")
        bytes_cum = r * len(clients) * 2 * p_tx if a.mode != "cl" else task.raw_train_bytes
        with open(log_path, "a") as f:
            f.write(f"{r},{m['accuracy']:.4f},{loss:.5f},{m['auc_roc']:.5f},"
                    f"{eps if eps == '' else f'{eps:.5f}'},{bytes_cum}\n")
        print(f"[{a.dataset} {a.mode} {a.partition} seed {a.seed}] round {r}: "
              f"val_acc {m['accuracy']:.2f}%" + (f"  eps {eps:.3f}" if a.dp else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------- centralized
    if a.mode == "cl":
        cw = class_weights(task.y_train[pool_idx]) if task.class_weighted else None
        loader = make_loader(task.train, pool_idx, a.batch_size, True, seed=a.seed * 1000)
        dp = ({"noise_multiplier": dp_cfg[0]["sigma"], "max_grad_norm": a.max_grad_norm}
              if a.dp else None)
        # one continuous training run (single optimiser); validation logged after every epoch
        final_model, _ = local_train(global_model, loader, R, a.lr, device, cw=cw, dp=dp,
                                     on_epoch_end=log_round)

    # ------------------------------------------------------------- federated (Flower)
    else:
        import flwr as fl
        from flwr.common import ndarrays_to_parameters

        final_holder = {}
        args_dict = vars(a)

        class HospitalClient(fl.client.NumPyClient):
            def __init__(self, cid):
                self.cid = cid

            def fit(self, parameters, config):
                c, d = clients[self.cid], dp_cfg[self.cid]
                rnd = int(config.get("server_round", 1))
                if os.environ.get("FLHC_TEST_FAIL_CLIENT") == str(self.cid):   # tests/ only
                    raise RuntimeError("deliberate test failure")
                if a.fp16:
                    parameters = fp16_roundtrip(parameters)           # FP16 download
                model, _ = make_model(task, argparse.Namespace(**args_dict), pretrained=False)
                set_weights(model, parameters)
                loader = make_loader(task.train, c["train"], a.batch_size, True,
                                     seed=a.seed * 100_000 + rnd * 100 + self.cid)
                cw = class_weights(task.y_train[c["train"]]) if task.class_weighted else None
                mu = float(config.get("proximal_mu", 0.0))
                dp = ({"noise_multiplier": d["sigma"], "max_grad_norm": a.max_grad_norm}
                      if a.dp else None)
                model, _ = local_train(model, loader, a.local_epochs, a.lr, device, mu=mu,
                                       global_weights=parameters, cw=cw, dp=dp)
                out = get_weights(model)
                if a.fp16:
                    out = fp16_roundtrip(out)                         # FP16 upload
                return out, len(c["train"]), {}

            def evaluate(self, parameters, config):
                return 0.0, 1, {}

        def client_fn(cid):
            c = HospitalClient(int(cid))
            return c.to_client() if hasattr(c, "to_client") else c

        def fit_config(server_round):
            return {"server_round": server_round}

        def evaluate_fn(server_round, parameters, config):
            if server_round == 0:
                return None
            set_weights(global_model, parameters)
            final_holder["w"] = [np.copy(x) for x in parameters]
            log_round(server_round, global_model)
            return 0.0, {}

        common = dict(accept_failures=False,   # a failed client aborts the run, never silently skipped
                      fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=len(clients),
                      min_available_clients=len(clients), evaluate_fn=evaluate_fn,
                      on_fit_config_fn=fit_config,
                      initial_parameters=ndarrays_to_parameters(init_weights))
        base_cls = fl.server.strategy.FedProx if a.mode == "fedprox" else fl.server.strategy.FedAvg
        final_holder["failures"] = 0

        class StrictStrategy(base_cls):
            """Counts client failures so that an incomplete round invalidates the run."""
            def aggregate_fit(self, server_round, results, failures):
                if failures:
                    final_holder["failures"] += len(failures)
                    print(f"ERROR: {len(failures)} client(s) failed in round {server_round}: "
                          f"{[repr(f) for f in failures][:1]}", flush=True)
                return super().aggregate_fit(server_round, results, failures)

        kwargs = {"proximal_mu": a.mu} if a.mode == "fedprox" else {}
        strategy = StrictStrategy(**kwargs, **common)
        res = {"num_cpus": a.client_cpus}
        if a.client_gpus:
            res["num_gpus"] = a.client_gpus
        ray_args = {"include_dashboard": False, "log_to_driver": False}
        if a.ray_cpus:
            ray_args["num_cpus"] = a.ray_cpus
        fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=len(clients),
            config=fl.server.ServerConfig(num_rounds=a.rounds), strategy=strategy,
            client_resources=res, ray_init_args=ray_args)
        last_round = int(open(log_path).read().strip().splitlines()[-1].split(",")[0]) \
            if len(open(log_path).read().strip().splitlines()) > 1 else 0
        if final_holder["failures"] or last_round != a.rounds or "w" not in final_holder:
            os.replace(log_path, log_path + ".INVALID")
            raise RuntimeError(f"{final_holder['failures']} client failure(s); training did not "
                               f"complete all {a.rounds} rounds cleanly. No results were written "
                               "and round_log.csv was renamed to round_log.csv.INVALID.")
        set_weights(global_model, final_holder["w"])
        final_model = global_model

    # ------------------------------------------------------------- final evaluation
    test_loader = make_loader(task.test, np.arange(len(task.test)), a.eval_batch_size, False)
    y, p, _ = predict(final_model, test_loader, device)
    metrics = binary_metrics(y, p)
    eps_final = (max(epsilon_spent(d["sigma"], d["q"], R * E * d["spe"], a.delta) for d in dp_cfg)
                 if a.dp else None)
    result = {"dataset": a.dataset, "mode": a.mode, "partition": a.partition, "seed": a.seed,
              "dp": a.dp, "epsilon": eps_final,
              "sigma_mean": float(np.mean([d["sigma"] for d in dp_cfg])) if a.dp else None,
              "fp16": a.fp16, "seconds": round(time.time() - t0, 1), **metrics}
    json.dump(result, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=2)

    if a.mode != "cl":
        with open(os.path.join(a.out, "client_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client", "n_train", "n_val", "val_acc", "val_auc"])
            for k, c in enumerate(clients):
                yl, pl, _ = predict(final_model, make_loader(task.train, c["val"], a.eval_batch_size, False), device)
                m = binary_metrics(yl, pl)
                w.writerow([k, len(c["train"]), len(c["val"]), round(m["accuracy"], 4),
                            "" if np.isnan(m["auc_roc"]) else round(m["auc_roc"], 5)])

    comm = {"n_params": n_params, "payload_fp32_bytes": p_fp32,
            "payload_tx_bytes_per_direction": p_tx, "fp16": a.fp16,
            "fl_total_bytes": (a.rounds * len(clients) * 2 * p_tx) if a.mode != "cl" else None,
            "cl_raw_data_bytes": task.raw_train_bytes}
    json.dump(comm, open(os.path.join(a.out, "comm.json"), "w"), indent=2)
    cfg = {**vars(a), "device": device, "weights_source": weights_source, "task_notes": task.notes,
           "dp_per_client": dp_cfg, "versions": versions()}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2, default=str)
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
