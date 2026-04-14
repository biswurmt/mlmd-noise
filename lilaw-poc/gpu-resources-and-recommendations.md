# GPU resources and recommendations for ML training (summary)

## System summary

- GPUs: 8 x NVIDIA GeForce RTX 2080 Ti (11,264 MiB each). Driver: 560.28.03. CUDA reported: 12.6.
- CPU: Intel Xeon Gold 6230, 80 logical CPUs (40 physical cores, 2 sockets).
- RAM: about 251 GiB total (about 190 GiB available).
- Disk: about 1.8 TiB root, about 1.3 TiB free.
- Docker: Docker version 20.10.22 is installed on the host.
- nvidia-container-cli / nvidia-container-toolkit: not detected in quick checks (install on host if you plan to run GPU containers).
- GPUs are currently idle (no running processes reported by nvidia-smi).

## High-level recommendations

- Run one experiment per GPU for many independent experiments (up to 8 concurrent jobs).
- For multi-GPU training: group GPUs by job size: 2 GPUs/job -> 4 concurrent jobs; 4 GPUs/job -> 2 concurrent jobs; 8 GPUs/job -> 1 job.
- Use mixed precision (AMP) to increase throughput and reduce memory usage on RTX 2080 Ti.
- Use gradient accumulation to emulate larger effective batch sizes when per-GPU memory is the bottleneck.
- Tune DataLoader (num_workers and pin_memory) and use fast local storage for datasets.
- Containerize with CUDA-compatible images matching the host driver/runtime (CUDA 12.x images are recommended given driver compatibility).
- Use a scheduler or launcher to avoid accidental GPU overcommit.

## Practical checks / ops to run

- Verify GPU access from inside containers (requires nvidia-container-toolkit or Docker GPU support):

  docker run --gpus all --rm nvidia/cuda:12.6-base nvidia-smi

- Check PCIe / NUMA topology:

  nvidia-smi topo -m

- Enable persistence mode to reduce driver initialization time:

  sudo nvidia-smi -pm 1

- Test power limit changes only after measuring and verifying:

  sudo nvidia-smi -i <gpu_id> -pl <watts>

## Concrete launch patterns and examples

- One experiment per GPU (bash):

  for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python train.py --config configs/exp_$i.yaml &
  done
  wait

- One GPU per Docker container example:

  docker run --gpus "device=0" -v /path/to/data:/data -w /workspace my-image:tag python train.py --data /data

- Four-GPU PyTorch distributed example:

  torchrun --nproc_per_node=4 train.py --config config.yaml

- Simple rotation scheduler (toy example):

  GPUS=(0 1 2 3 4 5 6 7)
  idx=0
  while read -r cmd; do
    gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$gpu $cmd &
    idx=$((idx+1))
  done < jobs_list.txt
  wait

Consider GNU parallel, ray[tune], or submitit for more robust local scheduling.

## Performance knobs

- Mixed precision: use torch.cuda.amp.autocast() and GradScaler.
- DataLoader: DataLoader(..., pin_memory=True, num_workers=N) where N depends on I/O and CPU; start with 8 and tune.
- Prefetch/caching datasets on local SSD where possible.
- Batch size guidance (tune empirically): ResNet-like CNNs: per-GPU batch ~64-128; Transformers/ViTs: per-GPU batch ~8-32.
- Use gradient accumulation to reach larger effective batch sizes when memory limited.

## Monitoring and logging

- Quick monitoring: nvidia-smi -l 1 or gpustat -cp (install gpustat with pip).
- Long-term: expose DCGM exporter to Prometheus + Grafana.
- Experiment tracking: wandb, TensorBoard, or mlflow.

## Quick start checklist

1) Confirm nvidia-smi shows 8 GPUs.
2) Enable persistence mode: sudo nvidia-smi -pm 1.
3) Verify Docker GPU access: docker run --gpus all --rm nvidia/cuda:12.6-base nvidia-smi. If it fails, install nvidia-container-toolkit.
4) Prepare a launcher that assigns CUDA_VISIBLE_DEVICES per process.
5) Add AMP to training code and tune num_workers + batch_size.
6) Run a small dry-run test and measure throughput.

## Notes & caveats

- RTX 2080 Ti has 11 GiB memory; plan batch sizes accordingly and use AMP.
- Driver reports CUDA 12.6; prefer matching container/runtime builds.
- Ensure user/permission and container runtime settings allow GPU access when running inside containers.

## Next steps I can do

- Create a bash scheduler script that dispatches commands to GPUs without overlap.
- Produce Docker run samples tailored to this repo.
- Inspect training code (train.py) here and recommend num_workers and batch sizes.
