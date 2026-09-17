# MiniMax-M3 two-node inference and graph operator profiling report

## Result

- Native vLLM two-node TP16 inference: passed with real weights and a 256-token chat response.
- vLLM-plugin-FL two-node TP16 inference: passed with both dummy and real weights. The final graph-mode service also returned a valid 256-token chat response after profiling.
- Native graph-mode operator profile: passed. The full measured 64 x (4096 input + 256 output) batch was traced.
- Plugin graph-mode operator profile: passed. The measured batch and the full profiling window completed; rank 0 emitted five consecutive 64-worker-step trace cycles.
- Every boolean conservation check in both extracted `summary.json` files is `true`.

## Runtime configuration

- Model: `/models/MiniMax-M3`
- Topology: 2 nodes, tensor parallel size 16 (8 GPUs per node)
- Main-node address: `10.5.3.1`
- GLOO/NCCL interface: `bond0.2208`
- RDMA HCAs: `mlx5_101` through `mlx5_108`
- `NCCL_IB_DISABLE=0`
- `NCCL_NVLS_ENABLE=0`
- CUDA graph capture sizes: `1,2,4,8,16,32,64`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- Prefix caching disabled; multimodal profiling skipped
- Profiling workload: 64 concurrent requests, 4096 input tokens/request, 256 output tokens/request

## Profiling scope and validity

| Metric | Native graph | Plugin graph |
|---|---:|---:|
| Warmup batch wall time | 39.406 s | 17.771 s |
| Profiled batch wall time | 16.713 s | 54.198 s |
| CPU operator events | 406,899 | 449,367 |
| Unique CPU operator names | 86 | 76 |
| GPU kernel events | 511,740 | 588,957 |
| Unique kernel names | 94 | 80 |
| Rank-0 kernel duration in trace | 16.537 s | 16.414 s |
| Operator/shape-matched kernel events | 8.87% | 8.90% |
| Operator/shape-matched kernel time | 45.80% | 45.45% |
| Conservation checks | all passed | all passed |

Both traces now cover their complete measured batches. The profiled wall times are still not a direct performance comparison: the plugin fix synchronously exports and gzip-compresses a trace every 64 worker steps, so its 54.198-second request wall time includes four intermediate export pauses. Kernel duration is the useful device-side comparison and differs by only -0.74% between plugin and native in these runs.

The original plugin failure was caused by one unbounded CUPTI recording cycle during long CUDA Graph replay. It first surfaced asynchronously in the NCCL watchdog as `CUDA error: unspecified launch failure`. Explicitly destroying and recreating the profiler bounded the data but could deadlock a real-weight distributed run during CUPTI re-attach. The final fix uses one attached profiler with an infinite PyTorch schedule: every 64 worker steps it executes `RECORD_AND_SAVE`, exports a trace, and clears accumulated events without detach/re-attach. The extractor natively merges the five rank-0 trace files.

## Largest operator/kernel relations

Native graph, by rank-0 kernel duration:

| Share | Calls | Operator | Kernel |
|---:|---:|---|---|
| 22.513% | 3,994 | `vllm::all_reduce` | NCCL bf16 ring LL all-reduce |
| 14.455% | 30,720 | unattributed graph replay | NCCL bf16 tree LL all-reduce |
| 12.254% | 29,184 | unattributed graph replay | `fused_moe_kernel` |
| 9.215% | 1,881 | unattributed graph replay | `_gqa_sparse_fwd_kernel` |
| 8.961% | 3,762 | `vllm::moe_forward_shared` | `fused_moe_kernel` |
| 4.005% | 34,680 | unattributed graph replay | FlashInfer fused add RMSNorm |
| 3.015% | 5,757 | `aten::mm` | NVJet SM90 GEMM 256x128 |
| 2.804% | 1,881 | `aten::mm` | SM80 XMMA GEMM |

Plugin graph, by rank-0 kernel duration in the complete five-cycle trace:

| Share | Calls | Operator | Kernel |
|---:|---:|---|---|
| 22.539% | 3,994 | `vllm::all_reduce` | NCCL bf16 ring LL all-reduce |
| 12.592% | 30,600 | unattributed graph replay | NCCL bf16 tree LL all-reduce |
| 12.396% | 29,070 | unattributed graph replay | `fused_moe_kernel` |
| 9.315% | 1,881 | unattributed graph replay | `_gqa_sparse_fwd_kernel` |
| 8.505% | 3,762 | `vllm::moe_forward_shared` | `fused_moe_kernel` |
| 4.010% | 34,560 | unattributed graph replay | FlashInfer fused add RMSNorm |
| 3.043% | 5,757 | `aten::mm` | NVJet SM90 GEMM 256x128 |
| 2.916% | 16,359 | unattributed graph replay | `_topk_index_partial_kernel` |

The dominant categories in both traces are tensor-parallel all-reduce, fused MoE, sparse GQA, GEMM, and fused normalization. Unattributed events are retained rather than dropped, as required by the extraction tool; most are CUDA Graph replay kernels for which PyTorch CPU metadata cannot be recovered.

## Plugin compatibility fixes

The editable plugin source was changed on both nodes so MiniMax-M3 can use the plugin implementation:

- Restrict the FlagGems fused-MoE fast path to SILU, allowing unsupported MiniMax activation to fall through.
- Forward MiniMax clamp/alpha/beta activation parameters.
- Implement `SWIGLUOAI_UNINTERLEAVE` with MiniMax's `swiglu_oai_split` operation.
- Add an opt-in `VLLM_FL_PROFILE_RANK0_ONLY=1` profiler guard for rank-0-only tracing.
- Add `VLLM_FL_PROFILE_SEGMENT_STEPS` cyclic trace flushing so long CUDA Graph profiles do not accumulate unbounded CUPTI state or repeatedly re-attach CUPTI.
- Make eager mode conditional so the same launch scripts can run eager by default or graph mode with `VLLM_ENFORCE_EAGER=0`.

## Dummy loading assessment

`--load-format dummy` is suitable for quickly validating two-node process startup, NCCL/GLOO wiring, plugin imports, model construction, CUDA graph capture, and profiler stability. It reduced the debug cycle enough to validate the cyclic 64-step trace flushing before the 59 real-weight shards were loaded. It cannot validate weight deserialization, model output quality, or real-weight numerical/performance behavior, so the final inference and operator capture were repeated with real weights.

## Local artifacts

- `native/`: complete native graph profile.
- `plugin_full/`: final complete plugin graph profile with five cyclic traces.
- `plugin/`: earlier bounded diagnostic profile, retained only for failure analysis.
