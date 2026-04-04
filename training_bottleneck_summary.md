# Training Bottleneck Summary

Date: 2026-03-26  
Source log: `monitor_20260326_194805.log`

## Scope

Analyzed the monitoring log collected during a run that covered roughly two full epochs of:

- `yolo train data=VOC.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False name=voc_profiling`

## Key Findings

- **Primary bottleneck: GPU compute**, not data loading from disk.
- Increasing workers to 16 was slower in prior experiments, and `cache=ram` did not improve throughput.
- Dataset location is already optimal (local NVMe on ext4).

## Evidence from the log

Training-active window detected:

- Start: `2026-03-26T19:48:12+03:00`
- End: `2026-03-26T19:52:03+03:00`
- Duration: `231s`

Snapshot counts:

- Total snapshots with GPU util: `1820`
- Training snapshots: `1742`
- Non-training snapshots: `78`

GPU utilization during training snapshots:

- Average: `92.3%`
- Median (p50): `96%`
- p90: `98%`
- Max: `100%`
- Low-util stalls were limited:
  - `<20%`: `44/1742` (`2.5%`)
  - `<40%`: `45/1742` (`2.6%`)

CPU observation:

- YOLO process + worker processes were active (`sum YOLO %CPU` ~`269%` average in training snapshots), but GPU remained highly occupied.

Disk observation:

- `iostat` was unavailable in this run (`sysstat` not installed), so no per-device queue/await metrics were captured.
- Given high sustained GPU utilization plus no benefit from `cache=ram`, disk I/O is unlikely to be the limiting factor.

## Conclusion

This run is **mostly compute-bound** (model forward/backward and augmentation compute), not input-I/O-bound.

## Safe speed-up options (without changing result-affecting hyperparameters)

- Keep `workers` at the empirically best value (likely 8-12, not 16).
- Reduce non-essential runtime overhead:
  - `verbose=False`
  - `plots=False`
  - less frequent checkpoint writes via `save_period`
- Keep dataset and `runs/` outputs on local NVMe.

## Measured optimizations (baseline: `batch=48 pretrained=False`, yolov8s VOC)

| Change | Train/epoch | Val/epoch | Notes |
|---|---|---|---|
| `deterministic=True` (baseline) | 1:20 | 10s | default config |
| `deterministic=False` | 1:16 | 9s | ~5% train, ~10% val speedup |
| TF32 matmul + `cudnn.benchmark=True` (isolated) | 1:20 | 9s | ~1.6% train, ~10% val vs no-TF32 |
| `deterministic=False` + TF32 + `cudnn.benchmark=True` (combined) | 1:15 | TBD | ~6% over determ=False alone, ~6% over TF32 alone |
| `torch.compile` (default mode) | No speedup | No speedup | 6+ epochs tested, no improvement |
| `torch.compile` (max-autotune mode) | No speedup | No speedup | Tested, no improvement despite VRAM headroom |

## Hardware context

- **GPU**: RTX 3090 (24GB VRAM, ~22GB available for training)
- **CPU**: i9-12900K (24 threads)
- **RAM**: 64GB
- **Storage**: Local NVMe SSD
- **PyTorch**: 2.7.1+cu126, CUDA 12.6, cuDNN 90501

## Optimization attempts (reverted)

### TF32 + cuDNN benchmark (reverted)
- **What**: `torch.backends.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.benchmark = True` in `init_seeds()`
- **Result**: Minimal gains (~1.6% train, ~10% val), not worth complexity
- **TF32 matmul benchmark**: 1.22x speedup on raw matmul, but limited impact in YOLO training

### torch.compile (reverted)
- **What**: Compiled model predict path with modes `default`, `reduce-overhead`, `max-autotune`
- **Result**: Zero speedup after 6+ epochs across all modes
- **Reason**: YOLO training already highly optimized (AMP + conv-heavy), little headroom for compile gains

## Final recommendation

**Only use `deterministic=False`** for ~5% training speedup. Other optimizations showed no meaningful benefit for this workload (yolov8s, RTX 3090, VOC dataset).

Command: `yolo train data=VOC.yaml model=yolov8s.pt imgsz=640 batch=48 epochs=65 pretrained=False deterministic=False`

