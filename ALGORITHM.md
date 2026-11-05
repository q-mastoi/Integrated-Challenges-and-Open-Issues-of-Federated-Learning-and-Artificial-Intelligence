# Algorithm 1: Federated training with optional FedProx and DP-SGD

Pseudocode for the manuscript (Section 5). It matches `run_experiment.py`.

```
Input:  K clients with local datasets D_1..D_K (Dirichlet(α) or IID partition)
        rounds R, local epochs E, learning rate η, batch size B
        proximal coefficient μ (μ = 0 for FedAvg)
        DP parameters: noise multiplier σ, clipping norm C, δ   (DP-SGD only)
Output: global model w_R, privacy budget ε

 1: Server initialises global weights w_0
 2: for each client k: q_k ← 1 / ⌈|D_k| / B⌉            ▷ Poisson sampling rate
 3: for round r = 1, ..., R do
 4:     Server sends w_{r-1} to all K clients
 5:     for each client k in parallel do
 6:         w ← w_{r-1}
 7:         for local epoch e = 1, ..., E do
 8:             for each mini-batch b sampled from D_k with rate q_k do
 9:                 L(w) ← (1/|b|) Σ_{i∈b} ℓ(w; x_i, y_i) + (μ/2) ‖w − w_{r-1}‖²
10:                 if DP-SGD then
11:                     g_i ← ∇ℓ(w; x_i, y_i) · min(1, C / ‖∇ℓ(w; x_i, y_i)‖₂)   ▷ per-sample clipping
12:                     g ← (Σ_{i∈b} g_i + N(0, σ²C² I)) / (q_k |D_k|)             ▷ Gaussian noise
13:                 else
14:                     g ← ∇L(w)
15:                 w ← AdamUpdate(w, g, η)
16:         Client k returns w_r^k and n_k = |D_k|
17:     Server aggregates w_r ← Σ_k (n_k / Σ_j n_j) · w_r^k                   ▷ FedAvg / FedProx
18:     Server evaluates w_r on the pooled validation set
19: ε ← max_k RDP-Accountant(σ, q_k, steps = R · E / q_k, δ)               ▷ Mironov (2017)
20: return w_R, ε
```

Notes
- FedProx (μ > 0) and DP-SGD are not combined, because per-sample clipping in Opacus does not apply to the proximal term.
- For the ε-sensitivity analysis (Table 3), σ is chosen per client as the smallest value for which line 19 gives ε ≤ ε_target over the full R · E epochs.
- With FP16 transmission, w is cast to float16 before lines 4 and 16 and back to float32 on receipt.
