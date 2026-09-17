# Runtime profile comparison

- Left: `plugin_graph`
- Right: `native_graph`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 112 | 117 | 105 | 7 | 12 |
| Kernel types | 93 | 114 | 60 | 33 | 54 |
| Known operator types | 46 | 49 | 36 | 10 | 13 |
| Known shape/dtype variants | 1334 | 1457 | 1225 | 109 | 232 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 10.542003 | 7.913440 | 0.750658 |
| Kernel events | 390624.000000 | 347278.000000 | 0.889034 |
| Kernel duration (us) | 6125271.191000 | 7156461.836000 | 1.168350 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 31.833558 | 28.239057 | -3.594501 |
| kernel_unified_attention | 13.680540 | 0.000000 | -13.680540 |
| mm_kernel_general_host_tma | 13.159865 | 0.000000 | -13.159865 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 5.542854 | 13.079322 | +7.536468 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 2.435140 | 11.393374 | +8.958234 |
| _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta | 5.358402 | 0.000000 | -5.358402 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 0.000000 | 4.590895 | +4.590895 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 0.000000 | 3.646270 | +3.646270 |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > >::Params) | 0.000000 | 3.491381 | +3.491381 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 3.204148 | 2.756346 | -0.447802 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 0.000000 | 2.931321 | +2.931321 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 0.922164 | 2.469626 | +1.547462 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 0.000000 | 2.211214 | +2.211214 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 1.957150 | 1.675412 | -0.281738 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 1.636420 | 1.401231 | -0.235189 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 13.707147 | 11.634881 | -2.072266 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 4.358472 | 12.110926 | +7.752454 |
| aten::mm | 11.565523 | 8.787767 | -2.777756 |
| symm_mem::two_shot_all_reduce_ | 2.435140 | 11.393374 | +8.958234 |
| vllm::unified_attention_with_output | 9.542867 | 0.000000 | -9.542867 |
| vllm::qwen_gdn_attention_core | 5.379497 | 4.617262 | -0.762235 |
| aten::sum | 0.000000 | 2.093107 | +2.093107 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 1.312816 | 1.125506 | -0.187310 |
| _vllm_fa3_C::fwd | 0.000000 | 1.034705 | +1.034705 |
| aten::mul | 0.460983 | 0.814261 | +0.353277 |
| _C::silu_and_mul | 0.000000 | 0.753830 | +0.753830 |
| aten::copy_ | 0.648419 | 0.557936 | -0.090482 |
| _moe_C::moe_align_block_size | 0.000000 | 0.625740 | +0.625740 |
| triton_per_fused_1 | 0.575913 | 0.492725 | -0.083187 |
| triton_poi_fused_add_4 | 0.528892 | 0.454885 | -0.074007 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
