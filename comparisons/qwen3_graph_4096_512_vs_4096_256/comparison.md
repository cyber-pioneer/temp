# Runtime profile comparison

- Left: `4096_512`
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
| Output tokens | 32768.000000 | 16384.000000 | 0.500000 |
| Batch wall time (s) | 13.685305 | 10.807638 | 0.789726 |
| Kernel events | 726240.000000 | 390624.000000 | 0.537872 |
| Kernel duration (us) | 9228845.981000 | 6107482.570000 | 0.661782 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 35.092109 | 31.670059 | -3.422050 |
| kernel_unified_attention | 12.107101 | 13.697070 | +1.589969 |
| mm_kernel_general_host_tma | 11.540288 | 13.212243 | +1.671956 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7.358026 | 5.382126 | -1.975900 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 4.477794 | 5.552138 | +1.074344 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 2.122891 | 3.208019 | +1.085128 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 2.649387 | 1.965211 | -0.684176 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 1.619253 | 2.442290 | +0.823038 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 1.302513 | 1.639753 | +0.337240 |
| nvjet_tst_64x8_64x16_1x1_h_bz_TNT | 1.530380 | 1.151871 | -0.378508 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.184055 | 0.901308 | -0.282747 |
| topk_gating_softmax_kernel | 1.164287 | 1.051579 | -0.112708 |
| _causal_conv1d_fwd_kernel | 0.659703 | 0.996482 | +0.336779 |
| nvjet_tst_320x64_64x4_2x1_v_bz_TNT | 0.967552 | 0.695974 | -0.271577 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 0.611598 | 0.923851 | +0.312253 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 9.091416 | 13.731796 | +4.640381 |
| aten::mm | 8.193317 | 11.608900 | +3.415583 |
| vllm::unified_attention_with_output | 6.318788 | 9.550115 | +3.231328 |
| vllm::qwen_gdn_attention_core | 3.567176 | 5.389907 | +1.822732 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 2.893377 | 4.371410 | +1.478034 |
| symm_mem::two_shot_all_reduce_ | 1.619253 | 2.442290 | +0.823038 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.869763 | 1.314478 | +0.444715 |
| aten::copy_ | 0.707350 | 0.650863 | -0.056486 |
| triton_per_fused_1 | 0.383572 | 0.579746 | +0.196174 |
| triton_poi_fused_add_4 | 0.347267 | 0.524772 | +0.177505 |
| aten::cat | 0.327836 | 0.495401 | +0.167565 |
| record_param_comms | 0.486480 | 0.372837 | -0.113643 |
| aten::mul | 0.302156 | 0.456324 | +0.154169 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_1 | 0.280878 | 0.424565 | +0.143687 |
| aten::argmax | 0.399651 | 0.313643 | -0.086009 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
