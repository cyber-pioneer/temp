# DeepSeek-V4-Flash 4096/256 four-scenario comparison

## Scope

All four runs used tensor parallel size 8, 64 concurrent requests, 4096 input
tokens per request, and 256 output tokens per request. Each run completed one
64-request warmup batch before `/start_profile`; only the following 64-request
batch is present in the runtime trace. Extraction uses rank 0 only. CUDA Graph
construction and capture are not profiled.

The native runs used an explicit empty `VLLM_PLUGINS` value. Their API-server
process environments and logs were checked during execution. Plugin runs loaded
and activated `fl`; their per-run FlagGems oplists were timestamp-validated and
moved into the corresponding `results` directories.

## Scenario summary

| Scenario | Profiled batch (s) | Kernel events | Kernel types | Kernel duration (us) | CPU op types | Known shape/dtype variants | Operator-kernel relations | Numbered IDs | Event mapping | Time mapping | Rank-0 trace (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 16.437 | 669,641 | 144 | 12,660,038.740 | 112 | 1,596 | 176 | 135 | 9.555% | 57.008% | 67.441 |
| plugin_eager | 78.350 | 689,411 | 155 | 13,493,531.931 | 110 | 2,882 | 157 | 77 | 76.787% | 83.759% | 241.966 |
| native_graph | 15.469 | 657,290 | 151 | 12,329,047.725 | 114 | 1,675 | 183 | 131 | 9.523% | 55.195% | 67.784 |
| native_eager | 53.293 | 602,756 | 155 | 30,737,180.086 | 109 | 2,759 | 157 | 72 | 73.449% | 92.776% | 223.116 |

Profiled batch time includes profiler overhead and is not an unprofiled throughput
benchmark. Kernel duration is the sum of rank-0 runtime kernel durations, not
end-to-end latency. Every conservation and classification check in all four
`summary.json` files is `true`.

## Relative timing

| Comparison | Profiled batch change | Kernel-duration change |
|---|---:|---:|
| plugin graph vs native graph | -5.885% | -2.614% |
| plugin eager vs native eager | -31.981% | +127.792% |
| plugin graph vs plugin eager | +376.677% | +6.584% |
| native graph vs native eager | +244.504% | +149.307% |

Changes are right relative to left. The two columns use different denominators and
must not be interpreted as equivalent latency measurements.

## Operator classification

| Scenario | aten | custom | runtime operator | torch compile | triton compiled | unattributed | unattributed nvjet | communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 50 | 28 | 0 | 1 | 1 | 88 | 4 | 4 |
| plugin_eager | 74 | 47 | 1 | 1 | 0 | 31 | 0 | 3 |
| native_graph | 56 | 28 | 0 | 1 | 1 | 89 | 4 | 4 |
| native_eager | 74 | 48 | 0 | 1 | 0 | 31 | 0 | 3 |

Counts are unique operator-kernel relations from `operator_list.csv`, not runtime
call counts. ATen relations are listed first. Pure communication relations remain
unnumbered and last. Missing attribution remains present as `operator_name=null`.

## Type and shape differences

Each cell is `left / right / intersection / left-only / right-only`.

| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |
|---|---|---|---|---|
| plugin graph vs native graph | 144 / 151 / 122 / 22 / 29 | 43 / 38 / 35 / 8 / 3 | 1596 / 1675 / 1359 / 237 / 316 | 112 / 114 / 109 / 3 / 5 |
| plugin eager vs native eager | 155 / 155 / 132 / 23 / 23 | 41 / 36 / 31 / 10 / 5 | 2882 / 2759 / 2097 / 785 / 662 | 110 / 109 / 103 / 7 / 6 |
| plugin graph vs plugin eager | 144 / 155 / 136 / 8 / 19 | 43 / 41 / 38 / 5 / 3 | 1596 / 2882 / 1275 / 321 / 1607 | 112 / 110 / 106 / 6 / 4 |
| native graph vs native eager | 151 / 155 / 140 / 11 / 15 | 38 / 36 / 33 / 5 / 3 | 1675 / 2759 / 1370 / 305 / 1389 | 114 / 109 / 106 / 8 / 3 |

Graph replay preserves physical kernel accounting but does not replay most original
CPU operator events. This is why graph runs have lower runtime logical shape
attribution than eager runs. Complete unions are retained in the pairwise CSV files.

## Top kernel time shares

### plugin_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,494,770.912 | 19.706% |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 24,360 | 1,576,722.203 | 12.454% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,277,130.063 | 10.088% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 973,979.466 | 7.693% |
| mhc_post_tilelang_kernel | 24,088 | 658,210.521 | 5.199% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 66,080 | 514,530.338 | 4.064% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 460,543.307 | 3.638% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 369,226.894 | 2.916% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 325,821.517 | 2.574% |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 1... | 2,752 | 275,265.046 | 2.174% |

### plugin_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 23,751 | 2,573,732.568 | 19.074% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,491,093.006 | 18.461% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,275,415.490 | 9.452% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 959,169.641 | 7.108% |
| mhc_post_tilelang_kernel | 24,088 | 660,255.144 | 4.893% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 458,322.857 | 3.397% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 64,428 | 448,847.143 | 3.326% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 366,788.535 | 2.718% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 324,656.195 | 2.406% |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 1... | 2,752 | 274,910.202 | 2.037% |

### native_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,475,368.432 | 20.078% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,270,223.667 | 10.303% |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 24,360 | 1,152,186.700 | 9.345% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 986,332.556 | 8.000% |
| mhc_post_tilelang_kernel | 24,088 | 658,454.888 | 5.341% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 66,080 | 520,301.450 | 4.220% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 460,010.822 | 3.731% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 12,384 | 388,094.829 | 3.148% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 369,464.369 | 2.997% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 326,949.037 | 2.652% |

### native_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 23,751 | 18,801,622.742 | 61.169% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,484,346.511 | 8.083% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,277,304.917 | 4.156% |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals, vllm:... | 1,305 | 1,008,954.055 | 3.283% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 963,166.244 | 3.134% |
| mhc_post_tilelang_kernel | 24,088 | 659,537.535 | 2.146% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 64,428 | 462,916.815 | 1.506% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 459,190.119 | 1.494% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 12,384 | 386,028.519 | 1.256% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 368,699.804 | 1.200% |

## Result locations

- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_rerun_20260908_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_rerun_20260908_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_rerun_20260908_baseline_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_rerun_20260908_baseline_eager_4096_256/`

Pairwise comparisons are under this report directory in `plugin_vs_native_graph`,
`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and
`native_graph_vs_eager`.
