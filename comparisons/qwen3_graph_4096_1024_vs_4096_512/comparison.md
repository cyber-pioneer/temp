# Runtime profile comparison

- Left: `4096_1024`
- Right: `4096_512`
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
| Output tokens | 65536.000000 | 32768.000000 | 0.500000 |
| Batch wall time (s) | 19.920416 | 13.685305 | 0.686999 |
| Kernel events | 1397472.000000 | 726240.000000 | 0.519681 |
| Kernel duration (us) | 16500445.833000 | 9228845.981000 | 0.559309 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 36.652866 | 35.092109 | -1.560757 |
| kernel_unified_attention | 10.427540 | 12.107101 | +1.679561 |
| mm_kernel_general_host_tma | 9.576920 | 11.540288 | +1.963368 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 8.361839 | 7.358026 | -1.003813 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 5.687238 | 4.477794 | -1.209444 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 3.345777 | 1.619253 | -1.726524 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 2.992200 | 2.649387 | -0.342813 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 1.187463 | 2.122891 | +0.935428 |
| nvjet_tst_64x8_64x16_1x1_h_bz_TNT | 1.710912 | 1.530380 | -0.180532 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.373284 | 1.184055 | -0.189229 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.972994 | 1.302513 | +0.329519 |
| topk_gating_softmax_kernel | 1.173618 | 1.164287 | -0.009331 |
| nvjet_tst_320x64_64x4_2x1_v_bz_TNT | 1.108820 | 0.967552 | -0.141269 |
| nvjet_tst_64x32_64x16_4x2_h_bz_TNT | 0.914735 | 0.804734 | -0.110001 |
| moe_align_block_size_stage2_vec | 0.861915 | 0.789567 | -0.072348 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 5.087294 | 9.091416 | +4.004121 |
| aten::mm | 5.148403 | 8.193317 | +3.044914 |
| vllm::unified_attention_with_output | 3.533521 | 6.318788 | +2.785267 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 3.918434 | 2.893377 | -1.025057 |
| vllm::qwen_gdn_attention_core | 1.995415 | 3.567176 | +1.571761 |
| symm_mem::two_shot_all_reduce_ | 3.345777 | 1.619253 | -1.726524 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.486881 | 0.869763 | +0.382882 |
| aten::copy_ | 0.704816 | 0.707350 | +0.002534 |
| record_param_comms | 0.541551 | 0.486480 | -0.055071 |
| aten::argmax | 0.438012 | 0.399651 | -0.038361 |
| triton_per_fused_1 | 0.214774 | 0.383572 | +0.168798 |
| triton_poi_fused_add_4 | 0.194423 | 0.347267 | +0.152844 |
| aten::cat | 0.183466 | 0.327836 | +0.144370 |
| aten::mul | 0.168815 | 0.302156 | +0.133341 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_1 | 0.157162 | 0.280878 | +0.123716 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
