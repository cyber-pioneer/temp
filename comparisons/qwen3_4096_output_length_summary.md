# Qwen 4096-token input: output-length comparison

## Scope

- Model: Qwen3.6-35B-A3B
- vLLM mode: graph or eager
- Tensor parallel size: 2
- Concurrency: 64
- Input tokens per request: 4096
- Output tokens per request: 1024, 512, or 256
- Complete warmup batch before profiling
- Rank-0 runtime trace only
- CUDA Graph construction excluded

Every conservation check passed for the 4096/512 graph and eager runs.
Percentages use total rank-0 runtime kernel duration as denominator.

## Graph mode

| Metric | 1024 | 512 | 256 |
|---|---:|---:|---:|
| CPU operator types | 112 | 112 | 112 |
| Kernel types | 93 | 93 | 93 |
| Kernel-attributed operator types | 46 | 46 | 46 |
| Known shape/dtype variants | 1334 | 1334 | 1334 |
| CPU operator events | 775212 | 585260 | 490284 |
| Kernel events | 1397472 | 726240 | 390624 |
| Kernel duration (us) | 16500445.833 | 9228845.981 | 6107482.570 |
| Profiled batch wall time (s) | 19.920416 | 13.685305 | 10.807638 |
| Shape-attributed kernel events | 5.807% | 9.412% | 15.861% |
| Shape-attributed kernel duration | 27.110% | 37.375% | 54.517% |

All three graph runs have identical CPU operator, kernel, attributed operator,
and known shape/dtype sets. Output length changes event counts and duration
shares, not physical kernel types.

Selected kernel shares:

| Kernel | 1024 | 512 | 256 |
|---|---:|---:|---:|
| fused_moe_kernel | 36.653% | 35.092% | 31.670% |
| kernel_unified_attention | 10.428% | 12.107% | 13.697% |
| mm_kernel_general_host_tma | 9.577% | 11.540% | 13.212% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 8.362% | 7.358% | 5.382% |

The graph-mode logical operator and shape percentages cover only events whose
runtime kernel retained a usable External id. Kernel inventory and kernel
duration percentages remain complete.

## Eager mode

| Metric | 1024 | 512 | 256 |
|---|---:|---:|---:|
| CPU operator types | 102 | 102 | 102 |
| Kernel types | 110 | 104 | 104 |
| Kernel-attributed operator types | 35 | 35 | 35 |
| Known shape/dtype variants | 3954 | 3863 | 3863 |
| CPU operator events | 15720819 | 8196935 | 4438855 |
| Kernel events | 3355706 | 1676706 | 896162 |
| Kernel duration (us) | 20888691.049 | 12849214.170 | 8241954.500 |
| Profiled batch wall time (s) | 355.902954 | 172.478111 | 94.719346 |
| Shape-attributed kernel events | 99.870% | 99.870% | 99.871% |
| Shape-attributed kernel duration | 99.948% | 99.956% | 99.960% |

The 512 and 256 eager runs have identical CPU operator, kernel, attributed
operator, and shape/dtype sets. Compared with 1024, both have ten fewer and
four additional low-share kernel implementations. Thirteen are nvjet GEMM
variants and one is a cuBLAS GEMV variant. Their aggregate shares are 0.050%
on the 1024 side and 0.073% on the 512 side.

Selected kernel shares:

| Kernel | 1024 | 512 | 256 |
|---|---:|---:|---:|
| fused_moe_kernel | 28.993% | 25.531% | 24.118% |
| kernel_unified_attention | 8.308% | 8.746% | 10.194% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 6.577% | 5.268% | 3.978% |
| cross_device_reduce_1stage | 6.347% | 3.589% | 1.263% |
| two_shot_all_reduce_kernel_inplace | 1.560% | 6.867% | 3.570% |

## Graph versus eager at 4096/512

| Metric | Graph | Eager | Intersection |
|---|---:|---:|---:|
| CPU operator types | 112 | 102 | 93 |
| Kernel types | 93 | 104 | 75 |
| Kernel-attributed operator types | 46 | 35 | 30 |
| Known shape/dtype variants | 1334 | 3863 | 1213 |

The profiled eager batch wall time is 12.60 times the graph value. This
includes substantially higher per-operator profiler overhead in eager mode
and is not a clean unprofiled throughput comparison.
