# Qwen3.6-35B-A3B 4096/256 four-scenario comparison

## Scope

All four runs used tensor parallel size 2, 64 concurrent requests, 4096 input
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
| plugin_graph | 10.803 | 391,150 | 96 | 6,165,191.780 | 112 | 1,377 | 125 | 81 | 16.271% | 54.300% | 30.452 |
| plugin_eager | 94.223 | 896,162 | 104 | 9,021,940.871 | 102 | 3,863 | 105 | 41 | 99.871% | 99.964% | 191.295 |
| native_graph | 8.026 | 347,278 | 114 | 6,909,075.250 | 117 | 1,457 | 144 | 82 | 16.962% | 58.499% | 30.280 |
| native_eager | 32.454 | 716,105 | 116 | 24,604,902.813 | 108 | 3,278 | 118 | 40 | 98.190% | 99.744% | 169.076 |

Profiled batch time includes profiler overhead and is not an unprofiled throughput
benchmark. Kernel duration is the sum of rank-0 runtime kernel durations, not
end-to-end latency. Every conservation and classification check in all four
`summary.json` files is `true`.

## Relative timing

| Comparison | Profiled batch change | Kernel-duration change |
|---|---:|---:|
| plugin graph vs native graph | -25.707% | +12.066% |
| plugin eager vs native eager | -65.557% | +172.723% |
| plugin graph vs plugin eager | +772.155% | +46.337% |
| native graph vs native eager | +304.345% | +256.124% |

Changes are right relative to left. The two columns use different denominators and
must not be interpreted as equivalent latency measurements.

## Operator classification

| Scenario | aten | custom | fused communication compute | runtime operator | torch compile | triton compiled | unattributed | unattributed nvjet | communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 48 | 13 | 2 | 0 | 3 | 15 | 33 | 7 | 4 |
| plugin_eager | 79 | 17 | 0 | 2 | 3 | 0 | 1 | 0 | 3 |
| native_graph | 59 | 14 | 2 | 0 | 3 | 15 | 34 | 13 | 4 |
| native_eager | 90 | 18 | 0 | 0 | 3 | 0 | 4 | 0 | 3 |

Counts are unique operator-kernel relations from `operator_list.csv`, not runtime
call counts. ATen relations are listed first. Pure communication relations remain
unnumbered and last. Missing attribution remains present as `operator_name=null`.

## Type and shape differences

Each cell is `left / right / intersection / left-only / right-only`.

| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |
|---|---|---|---|---|
| plugin graph vs native graph | 96 / 114 / 58 / 38 / 56 | 46 / 49 / 36 / 10 / 13 | 1377 / 1457 / 773 / 604 / 684 | 112 / 117 / 105 / 7 / 12 |
| plugin eager vs native eager | 104 / 116 / 65 / 39 / 51 | 35 / 36 / 24 / 11 / 12 | 3863 / 3278 / 2750 / 1113 / 528 | 102 / 108 / 96 / 6 / 12 |
| plugin graph vs plugin eager | 96 / 104 / 72 / 24 / 32 | 46 / 35 / 30 / 16 / 5 | 1377 / 3863 / 772 / 605 / 3091 | 112 / 102 / 93 / 19 / 9 |
| native graph vs native eager | 114 / 116 / 94 / 20 / 22 | 49 / 36 / 33 / 16 / 3 | 1457 / 3278 / 1284 / 173 / 1994 | 117 / 108 / 99 / 18 / 9 |

Graph replay preserves physical kernel accounting but does not replay most original
CPU operator events. This is why graph runs have lower runtime logical shape
attribution than eager runs. Complete unions are retained in the pairwise CSV files.

## Top kernel time shares

### plugin_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 1,986,835.963 | 32.227% |
| kernel_unified_attention | 2,890 | 838,348.547 | 13.598% |
| mm_kernel_general_host_tma | 31,790 | 807,045.369 | 13.090% |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::tr... | 11,560 | 339,126.187 | 5.501% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,650 | 328,072.502 | 5.321% |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective:... | 1,020 | 198,091.804 | 3.213% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 1,353 | 148,706.223 | 2.412% |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 19,760 | 119,888.094 | 1.945% |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 8,670 | 100,064.276 | 1.623% |
| nvjet_tst_64x8_64x16_1x1_h_bz_TNT | 10,120 | 70,689.459 | 1.147% |

### plugin_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 1,969,607.305 | 21.831% |
| kernel_unified_attention | 2,890 | 841,909.135 | 9.332% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 2,673 | 835,060.038 | 9.256% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat... | 104,917 | 714,466.102 | 7.919% |
| mul_broadcast_2d_kernel | 75,718 | 433,173.807 | 4.801% |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm:... | 20,736 | 355,256.134 | 3.938% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,680 | 327,641.380 | 3.632% |
| pow_func_tensor_scalar_kernel_rank_1 | 37,859 | 306,012.773 | 3.392% |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::Tenso... | 60,979 | 297,649.623 | 3.299% |
| add_func_kernel_rank_1 | 41,038 | 270,356.302 | 2.997% |

### native_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 2,023,655.368 | 29.290% |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::tr... | 11,560 | 783,957.799 | 11.347% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 1,353 | 701,988.765 | 10.160% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,680 | 328,606.496 | 4.756% |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 950 | 260,854.713 | 3.776% |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Collective... | 2,560 | 250,064.214 | 3.619% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 11,560 | 210,266.420 | 3.043% |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective:... | 990 | 197,211.971 | 2.854% |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 3,830 | 176,742.344 | 2.558% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::Binary... | 11,560 | 158,251.176 | 2.290% |

### native_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm:... | 21,060 | 17,016,524.530 | 69.159% |
| fused_moe_kernel | 23,440 | 1,982,069.837 | 8.056% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat... | 80,039 | 551,964.987 | 2.243% |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::Binary... | 58,731 | 439,325.969 | 1.786% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 2,673 | 363,947.795 | 1.479% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,800 | 329,391.000 | 1.339% |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 950 | 259,218.362 | 1.054% |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::Tenso... | 53,033 | 251,310.078 | 1.021% |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Collective... | 2,600 | 250,109.163 | 1.017% |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, flo... | 29,593 | 214,472.741 | 0.872% |

## Result locations

- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_rerun_20260908_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_rerun_20260908_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_rerun_20260908_baseline_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_rerun_20260908_baseline_eager_4096_256/`

Pairwise comparisons are under this report directory in `plugin_vs_native_graph`,
`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and
`native_graph_vs_eager`.
