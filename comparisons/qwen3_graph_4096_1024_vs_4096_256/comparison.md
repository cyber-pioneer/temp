# Runtime profile comparison

- Left: `4096_1024`
- Right: `4096_256`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 112 | 112 | 112 | 0 | 0 |
| Kernel types | 93 | 93 | 93 | 0 | 0 |
| Known operator types | 46 | 46 | 46 | 0 | 0 |
| Known shape/dtype variants | 1334 | 1334 | 1334 | 0 | 0 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 65536.000000 | 16384.000000 | 0.250000 |
| Batch wall time (s) | 19.920416 | 10.807638 | 0.542541 |
| Kernel events | 1397472.000000 | 390624.000000 | 0.279522 |
| Kernel duration (us) | 16500445.833000 | 6107482.570000 | 0.370140 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 36.652866 | 31.670059 | -4.982807 |
| kernel_unified_attention | 10.427540 | 13.697070 | +3.269530 |
| mm_kernel_general_host_tma | 9.576920 | 13.212243 | +3.635323 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 8.361839 | 5.382126 | -2.979713 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 5.687238 | 5.552138 | -0.135100 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 3.345777 | 2.442290 | -0.903486 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 1.187463 | 3.208019 | +2.020556 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 2.992200 | 1.965211 | -1.026989 |
| nvjet_tst_64x8_64x16_1x1_h_bz_TNT | 1.710912 | 1.151871 | -0.559041 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.972994 | 1.639753 | +0.666759 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.373284 | 0.901308 | -0.471976 |
| topk_gating_softmax_kernel | 1.173618 | 1.051579 | -0.122039 |
| nvjet_tst_320x64_64x4_2x1_v_bz_TNT | 1.108820 | 0.695974 | -0.412846 |
| _causal_conv1d_fwd_kernel | 0.369030 | 0.996482 | +0.627451 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 0.341823 | 0.923851 | +0.582028 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 5.087294 | 13.731796 | +8.644502 |
| aten::mm | 5.148403 | 11.608900 | +6.460497 |
| vllm::unified_attention_with_output | 3.533521 | 9.550115 | +6.016595 |
| vllm::qwen_gdn_attention_core | 1.995415 | 5.389907 | +3.394493 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 3.918434 | 4.371410 | +0.452976 |
| symm_mem::two_shot_all_reduce_ | 3.345777 | 2.442290 | -0.903486 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.486881 | 1.314478 | +0.827597 |
| aten::copy_ | 0.704816 | 0.650863 | -0.053952 |
| triton_per_fused_1 | 0.214774 | 0.579746 | +0.364972 |
| record_param_comms | 0.541551 | 0.372837 | -0.168714 |
| triton_poi_fused_add_4 | 0.194423 | 0.524772 | +0.330349 |
| aten::cat | 0.183466 | 0.495401 | +0.311935 |
| aten::mul | 0.168815 | 0.456324 | +0.287510 |
| aten::argmax | 0.438012 | 0.313643 | -0.124370 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_1 | 0.157162 | 0.424565 | +0.267403 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
