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
| plugin_graph | 10.542 | 390,624 | 93 | 6,125,271.191 | 112 | 1,334 | 121 | 81 | 15.861% | 54.404% | 29.969 |
| plugin_eager | 95.815 | 896,162 | 104 | 8,838,866.189 | 102 | 3,863 | 105 | 46 | 99.871% | 99.963% | 191.493 |
| native_graph | 7.913 | 347,278 | 114 | 7,156,461.836 | 117 | 1,457 | 144 | 78 | 16.962% | 60.006% | 30.296 |
| native_eager | 32.210 | 707,105 | 118 | 7,640,940.599 | 108 | 3,212 | 120 | 45 | 98.192% | 99.179% | 160.660 |

Profiled batch time includes profiler overhead and is not an unprofiled throughput
benchmark. Kernel duration is the sum of rank-0 runtime kernel durations, not
end-to-end latency. Every conservation and classification check in all four
`summary.json` files is `true`.

## Relative timing

| Comparison | Profiled batch change | Kernel-duration change |
|---|---:|---:|
| plugin graph vs native graph | -24.934% | +16.835% |
| plugin eager vs native eager | -66.383% | -13.553% |
| plugin graph vs plugin eager | +808.886% | +44.302% |
| native graph vs native eager | +307.035% | +6.770% |

Changes are right relative to left. The two columns use different denominators and
must not be interpreted as equivalent latency measurements.

## Operator classification

| Scenario | aten | custom | fused communication compute | runtime operator | torch compile | triton compiled | unattributed | unattributed nvjet | communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 45 | 12 | 2 | 0 | 3 | 15 | 33 | 7 | 4 |
| plugin_eager | 79 | 17 | 0 | 2 | 3 | 0 | 1 | 0 | 3 |
| native_graph | 59 | 14 | 2 | 0 | 3 | 15 | 34 | 13 | 4 |
| native_eager | 90 | 20 | 0 | 0 | 3 | 0 | 4 | 0 | 3 |

Counts are unique operator-kernel relations from `operator_list.csv`, not runtime
call counts. ATen relations are listed first. Pure communication relations are
numbered and listed last. Missing attribution remains present as `operator_name=null`.

## Type and shape differences

Each cell is `left / right / intersection / left-only / right-only`.

| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |
|---|---|---|---|---|
| plugin graph vs native graph | 93 / 114 / 60 / 33 / 54 | 46 / 49 / 36 / 10 / 13 | 1334 / 1457 / 1225 / 109 / 232 | 112 / 117 / 105 / 7 / 12 |
| plugin eager vs native eager | 104 / 118 / 57 / 47 / 61 | 35 / 36 / 24 / 11 / 12 | 3863 / 3212 / 960 / 2903 / 2252 | 102 / 108 / 96 / 6 / 12 |
| plugin graph vs plugin eager | 93 / 104 / 75 / 18 / 29 | 46 / 35 / 30 / 16 / 5 | 1334 / 3863 / 1213 / 121 / 2650 | 112 / 102 / 93 / 19 / 9 |
| native graph vs native eager | 114 / 118 / 87 / 27 / 31 | 49 / 36 / 33 / 16 / 3 | 1457 / 3212 / 784 / 673 / 2428 | 117 / 108 / 99 / 18 / 9 |

Graph replay preserves physical kernel accounting but does not replay most original
CPU operator events. This is why graph runs have lower runtime logical shape
attribution than eager runs. Complete unions are retained in the pairwise CSV files.

## Top kernel time shares

### plugin_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 1,949,891.769 | 31.834% |
| kernel_unified_attention | 2,890 | 837,970.200 | 13.681% |
| mm_kernel_general_host_tma | 31,790 | 806,077.443 | 13.160% |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::tr... | 11,560 | 339,514.814 | 5.543% |
| _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta | 7,680 | 328,216.682 | 5.358% |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective:... | 990 | 196,262.777 | 3.204% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 1,353 | 149,158.942 | 2.435% |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 19,760 | 119,880.716 | 1.957% |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 8,670 | 100,235.133 | 1.636% |
| nvjet_tst_64x8_64x16_1x1_h_bz_TNT | 10,120 | 70,600.351 | 1.153% |

### plugin_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 1,979,115.708 | 22.391% |
| kernel_unified_attention | 2,890 | 839,936.017 | 9.503% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat... | 104,917 | 714,776.859 | 8.087% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 2,673 | 658,451.661 | 7.450% |
| mul_broadcast_2d_kernel | 75,718 | 433,267.132 | 4.902% |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm:... | 20,736 | 342,565.117 | 3.876% |
| _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta | 7,680 | 327,109.535 | 3.701% |
| pow_func_tensor_scalar_kernel_rank_1 | 37,859 | 305,900.866 | 3.461% |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::Tenso... | 60,979 | 297,710.353 | 3.368% |
| add_func_kernel_rank_1 | 41,038 | 270,382.080 | 3.059% |

### native_graph

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 2,020,917.338 | 28.239% |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::tr... | 11,560 | 936,016.673 | 13.079% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 1,353 | 815,362.472 | 11.393% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,680 | 328,545.636 | 4.591% |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 950 | 260,943.890 | 3.646% |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Collective... | 2,560 | 249,859.345 | 3.491% |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapp... | 11,560 | 209,778.899 | 2.931% |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective:... | 990 | 197,256.857 | 2.756% |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 3,830 | 176,737.868 | 2.470% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::Binary... | 11,560 | 158,244.661 | 2.211% |

### native_eager

| Kernel | Calls | Kernel time (us) | Share |
|---|---:|---:|---:|
| fused_moe_kernel | 23,120 | 1,985,876.642 | 25.990% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterat... | 78,907 | 549,003.028 | 7.185% |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::Binary... | 58,287 | 438,007.974 | 5.732% |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7,680 | 327,913.694 | 4.292% |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat1... | 2,592 | 293,086.139 | 3.836% |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm:... | 20,817 | 283,768.931 | 3.714% |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 960 | 265,221.504 | 3.471% |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::Tenso... | 52,309 | 250,619.845 | 3.280% |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::Collective... | 2,560 | 249,088.390 | 3.260% |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, flo... | 29,189 | 213,546.714 | 2.795% |

## Result locations

- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_refactor_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_refactor_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_refactor_native_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_refactor_native_eager_4096_256/`

Pairwise comparisons are under this report directory in `plugin_vs_native_graph`,
`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and
`native_graph_vs_eager`.
