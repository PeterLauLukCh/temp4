# Connect-3 AlphaZero critic: A100 run bundle

This repository contains only the code needed to train and verify the
OpenSpiel 4x4 Connect-3 critic:

```text
connect_four(rows=4,columns=4,x_in_row=3)
```

It does not contain the paper, theory code, 2FFS, MCTS-BAI, or unrelated
benchmarks. OpenSpiel and its required C++ dependencies are downloaded at
fixed revisions through `gh-proxy.com`.

## Fixed versions

- OpenSpiel: `112b77704631fc2ce7ad8e4581f6ca09798ce15a`
- JAX: `jax[cuda12]==0.8.1`
- AlphaZero Python dependencies: see
  `benchmark/connect3/requirements-alpha-zero.txt`

The setup expects Ubuntu/Debian, one NVIDIA A100, an NVIDIA driver supporting
CUDA 12, and Python 3.11--3.13.

## 1. Download on the GPU node

```bash
wget -O temp4-main.zip \
  'https://gh-proxy.com/https://github.com/PeterLauLukCh/temp4/archive/refs/heads/main.zip'
unzip temp4-main.zip
cd temp4-main
```

Do not reuse a directory containing earlier runs.

## 2. Inspect the node and create the environment

```bash
bash scripts/inspect_node.sh
bash scripts/bootstrap_gpu.sh
```

`bootstrap_gpu.sh`:

1. installs the required Ubuntu build packages;
2. creates the project-local `.venv`;
3. downloads the fixed OpenSpiel source and build dependencies through
   `gh-proxy.com`;
4. builds OpenSpiel;
5. installs `jax[cuda12]==0.8.1` and the pinned AlphaZero dependencies;
6. runs `pip check`; and
7. requires `jax.default_backend() == "gpu"`.

If system packages are already installed and sudo is unavailable:

```bash
SKIP_APT=1 bash scripts/bootstrap_gpu.sh
```

The script refuses to reuse an existing `.venv`. Use a fresh checkout after a
partial or failed installation.

## 3. GPU smoke test

```bash
bash scripts/run_gpu_smoke.sh
```

Every invocation uses a new UTC-stamped `runs/connect3/gpu-smoke-*` directory.
It checks that the final smoke checkpoint, `learner.jsonl`, and actor game log
were written. The selected path is recorded in
`runs/connect3/latest_gpu_smoke.txt`.

## 4. Launch the pre-specified run0

Only launch run0 after the smoke script succeeds:

```bash
bash scripts/launch_run0.sh
```

The formal configuration is fixed to:

- ResNet width 64, depth 3
- 4 actors, 0 evaluators
- 64 MCTS simulations per move
- batch size 512
- replay buffer 8192, replay reuse 4
- 200 learner steps
- checkpoint frequency 20

The process is launched with `nohup`. Important files:

```text
runs/connect3/az_resnet_w64_d3_s64_run0.pid
runs/connect3/az_resnet_w64_d3_s64_run0.console.log
runs/connect3/az_resnet_w64_d3_s64_run0.launch.txt
runs/connect3/az_resnet_w64_d3_s64_run0.resource.csv
runs/connect3/az_resnet_w64_d3_s64_run0.resource.log
runs/connect3/az_resnet_w64_d3_s64_run0/
```

The launch script refuses to overwrite any of these paths.

## 5. Monitor

```bash
bash scripts/status_run0.sh
watch -n 30 bash scripts/status_run0.sh
```

Low GPU utilization can be normal because self-play MCTS is CPU-bound. Do not
change the formal configuration solely because utilization is low.

Successful completion creates `checkpoint-200`. The resource monitor exits
after the training PID exits and records GPU utilization/memory plus learner
and actor process CPU/RAM usage every 30 seconds.

## 6. Verify checkpoints and extract training states

After run0 finishes:

```bash
bash scripts/verify_and_extract.sh
```

This verifies the pre-specified checkpoints 20, 100, and 200 using plies
6--10 and at most 1000 mirror-deduplicated states, then creates:

```text
runs/connect3/az_resnet_w64_d3_s64_run0/verification-checkpoint-20.json
runs/connect3/az_resnet_w64_d3_s64_run0/verification-checkpoint-100.json
runs/connect3/az_resnet_w64_d3_s64_run0/verification-checkpoint-200.json
runs/connect3/az_resnet_w64_d3_s64_run0/training_states.json
```

The verification sample is only a critic sanity check, not the final benchmark
test split. A later calibration/test splitter must reject any root whose
canonical descendant closure intersects `training_states.json`, thereby
excluding training-state mirrors, transpositions, and descendants.
