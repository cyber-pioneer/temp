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
| plugin_graph | 16.326 | 669,641 | 144 | 12,908,520.800 | 112 | 1,596 | 176 | 99 | 9.555% | 57.863% | 67.408 |
| plugin_eager | 78.677 | 689,411 | 155 | 11,998,409.944 | 110 | 2,882 | 157 | 72 | 76.787% | 81.775% | 241.795 |
| native_graph | 15.603 | 657,290 | 151 | 12,956,103.829 | 114 | 1,675 | 183 | 86 | 9.523% | 57.856% | 67.759 |
| native_eager | 53.258 | 602,756 | 155 | 29,810,023.945 | 109 | 2,759 | 157 | 67 | 73.449% | 92.558% | 223.753 |

Profiled batch time includes profiler overhead and is not an unprofiled throughput
benchmark. Kernel duration is the sum of rank-0 runtime kernel durations, not
end-to-end latency. Every conservation and classification check in all four
`summary.json` files is `true`.

## Relative timing

| Comparison | Profiled batch change | Kernel-duration change |
|---|---:|---:|
| plugin graph vs native graph | -4.426% | +0.369% |
| plugin eager vs native eager | -32.308% | +148.450% |
| plugin graph vs plugin eager | +381.909% | -7.050% |
| native graph vs native eager | +241.322% | +130.085% |

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
call counts. ATen relations are listed first. Pure communication relations are
numbered and listed last. Missing attribution remains present as `operator_name=null`.

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
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,494,194.387 | 19.322% |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 24,360 | 1,823,737.679 | 14.128% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,277,267.339 | 9.895% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 976,606.455 | 7.566% |
| mhc_post_tilelang_kernel | 24,088 | 657,642.738 | 5.095% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 66,080 | 514,458.832 | 3.985% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 460,600.124 | 3.568% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 369,905.460 | 2.866% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 326,035.685 | 2.526% |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 1... | 2,752 | 274,888.565 | 2.130% |

### plugin_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,491,689.759 | 20.767% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,275,718.173 | 10.632% |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 23,751 | 1,260,547.443 | 10.506% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 959,714.558 | 7.999% |
| mhc_post_tilelang_kernel | 24,088 | 659,437.456 | 5.496% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 457,704.356 | 3.815% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 64,428 | 446,669.347 | 3.723% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 366,429.161 | 3.054% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 324,640.947 | 2.706% |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 1... | 2,752 | 274,839.737 | 2.291% |

### native_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,476,774.557 | 19.117% |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 24,360 | 1,773,710.154 | 13.690% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,270,692.204 | 9.808% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 985,925.331 | 7.610% |
| mhc_post_tilelang_kernel | 24,088 | 657,791.701 | 5.077% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 66,080 | 520,499.182 | 4.017% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 459,993.086 | 3.550% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 12,384 | 388,156.852 | 2.996% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 369,478.995 | 2.852% |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10:... | 1,419 | 326,832.700 | 2.523% |

### native_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigne... | 23,751 | 17,842,034.081 | 59.852% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 2,838 | 2,487,951.439 | 8.346% |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTem... | 1,419 | 1,278,188.586 | 4.288% |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals, vllm:... | 1,305 | 1,042,320.849 | 3.497% |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 28147497671... | 21,844 | 961,753.250 | 3.226% |
| mhc_post_tilelang_kernel | 24,088 | 659,157.819 | 2.211% |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10:... | 64,428 | 461,270.003 | 1.547% |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 24,768 | 459,214.130 | 1.540% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 12,384 | 385,914.300 | 1.295% |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::... | 12,341 | 368,596.247 | 1.236% |

## Result locations

- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_refactor_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_refactor_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_refactor_native_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/deepseek_v4_flash_refactor_native_eager_4096_256/`

Pairwise comparisons are under this report directory in `plugin_vs_native_graph`,
`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and
`native_graph_vs_eager`.
