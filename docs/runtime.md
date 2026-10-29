# Local model runtimes

The controller, scorer and dashboard do not install Torch, Transformers or vLLM.
Use a dedicated inference environment; install the controller with `pip install .`
on the Linux GPU host. The supervisor requires `nvidia-smi`, and vLLM profiles
require Docker with NVIDIA GPU support. It waits for selected GPUs to have no
compute processes and at most 256 MiB used memory. It only cleans up its own worker
and container. Locks coordinate a workspace; unrelated jobs can race with the idle
check, so reserve the GPU separately when sharing a host.

## MiniCPM-V 4.7 official INT4 AWQ

Reference checkpoint:
`openbmb/MiniCPM-V-4.7-35B-A3B-AWQ` at
`9832dea29707a60ba3c2001b6ad5533212d150cd`.
The routed MoE experts use packed INT4 W4A16 group-128 compressed-tensors weights;
vision, attention and other excluded modules retain high precision. The adapter
checks 40 expert layers and the actual quantized computation modules after loading.

```bash
python -m pip install 'huggingface-hub[cli]'
hf download openbmb/MiniCPM-V-4.7-35B-A3B-AWQ \
  --revision 9832dea29707a60ba3c2001b6ad5533212d150cd --local-dir models/minicpm47-awq
python scripts/prepare_minicpm47.py --source models/minicpm47-awq --output models/minicpm47-runtime
docker build -t vlmbench-minicpm47:local runtimes/minicpm47
```

The pinned export names legacy custom image/video processor aliases. The preparation
script verifies the known processor hashes and creates a separate copy using native
Transformers aliases, removing `auto_map` without changing tensor bytes. It copies
weights, so reserve another approximately 23 GB; source files stay intact. Its manifest
records both source/runtime hashes. Review downloaded code before enabling remote code;
this profile uses `trust_remote_code=false`.

The Docker recipe pins the original vLLM 0.32.0 base digest and Transformers 5.18.0,
retaining the base dependencies; it includes the NCCL 2.29.7 and telemetry dependency
adjustments used in the archived runtime. Builds download packages and must pass
`pip check`. Capture the resulting image ID and `pip freeze --all` for each run; the
published recipe is not a promise of a byte-identical rebuild of the archived image.
It targets Linux amd64 CUDA GPUs, not Jetson/Orin.

Copy `configs/models.example.json` to `configs/models.local.json`, set `path` to the
absolute runtime-view directory, set `docker_image` to your built image ID, and choose
idle physical `gpus`. `host: auto` resolves to the current host. The example is a
conservative 16,384-context / 128-output-token smoke profile. The archived full run
used **131,072 context, `max_new_tokens: null`** (remaining context until EOS),
`thinking: false`, 900-second request deadlines, 16x visual compression, at most four
slices for a single image and one per image for multiple images. To reproduce that
budget, change both fields explicitly before a full run. Check memory capacity; the
archived run used a 48 GiB GPU, with approximately 38.41 GiB peak including KV cache.

```bash
vlmbench run --suite data/public-vlm-v2 --models configs/models.local.json \
  --model minicpm47-awq --workspace runs/smoke-awq --run-id smoke --view smoke
```

Use a new workspace for the full run. Snapshot configuration, raw responses, worker
logs, GPU monitoring, timeout records and release evidence remain under that workspace.
Loading and warmup are excluded from formal per-request timing; they currently have
no separate deadline. A load failure returns nonzero with diagnostics. Do not count
an unsuccessful load as a valid AWQ evaluation.

## Other adapters

`adapters.py` and `additional_adapters.py` include Transformers MiniCPM, native Qwen
and vLLM multimodal adapters. Each needs a compatible separately prepared runtime,
verified checkpoint and processor configuration. For Transformers, set the backend's
`python` to an absolute interpreter path. Do not assume an AWQ profile is interchangeable
with BF16, or that a single framework version supports every model. API models can
use `vlmbench api-run` without any local CUDA dependencies.
