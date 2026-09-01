# Runtime profile comparison

- Left: `native_graph`
- Right: `native_eager`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 117 | 108 | 99 | 18 | 9 |
| Kernel types | 114 | 123 | 90 | 24 | 33 |
| Known operator types | 49 | 36 | 33 | 16 | 3 |
| Known shape/dtype variants | 1457 | 3349 | 820 | 637 | 2529 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 7.732620 | 32.799106 | 4.241655 |
| Kernel events | 347278.000000 | 716851.000000 | 2.064199 |
| Kernel duration (us) | 6522629.903000 | 11016806.643000 | 1.689013 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.007264 | 32.342589 | +31.335324 |
| fused_moe_kernel | 31.006077 | 18.176157 | -12.829920 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 10.262445 | 0.000000 | -10.262445 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 6.897359 | 2.943472 | -3.953888 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 5.035743 | 2.980190 | -2.055553 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 0.199535 | 5.002237 | +4.802702 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 4.000421 | 2.338675 | -1.661747 |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 0.000000 | 3.982946 | +3.982946 |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > >::Params) | 3.832890 | 2.263470 | -1.569420 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 3.223145 | 1.907483 | -1.315662 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 3.013938 | 1.804050 | -1.209888 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 2.710106 | 1.580536 | -1.129570 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 2.431551 | 0.838976 | -1.592575 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 0.000000 | 2.273878 | +2.273878 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4>) | 0.000000 | 1.944367 | +1.944367 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| _C_custom_ar::all_reduce | 0.132424 | 32.342589 | +32.210165 |
| vllm::moe_forward_shared | 12.785310 | 18.176157 | +5.390846 |
| aten::mm | 9.642638 | 9.544375 | -0.098263 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 9.224341 | 0.000000 | -9.224341 |
| aten::copy_ | 0.612113 | 9.102034 | +8.489921 |
| symm_mem::two_shot_all_reduce_ | 6.897359 | 2.943472 | -3.953888 |
| vllm::qwen_gdn_attention_core | 5.054888 | 6.267845 | +1.212957 |
| aten::mul | 0.893560 | 4.963714 | +4.070154 |
| aten::add | 0.012095 | 3.088885 | +3.076790 |
| _vllm_fa3_C::fwd | 1.134983 | 3.043474 | +1.908491 |
| aten::sum | 2.302456 | 1.907483 | -0.394973 |
| aten::mean | 0.000000 | 1.944367 | +1.944367 |
| aten::pow | 0.000000 | 1.514572 | +1.514572 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 1.234686 | 0.000000 | -1.234686 |
| _C::silu_and_mul | 0.827095 | 0.976338 | +0.149243 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
