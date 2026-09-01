# Runtime profile comparison

- Left: `plugin_eager`
- Right: `native_eager`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 102 | 108 | 96 | 6 | 12 |
| Kernel types | 104 | 123 | 61 | 43 | 62 |
| Known operator types | 35 | 36 | 24 | 11 | 12 |
| Known shape/dtype variants | 3863 | 3349 | 1025 | 2838 | 2324 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 93.152467 | 32.799106 | 0.352101 |
| Kernel events | 896162.000000 | 716851.000000 | 0.799912 |
| Kernel duration (us) | 8207711.519000 | 11016806.643000 | 1.342251 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.273635 | 32.342589 | +31.068954 |
| fused_moe_kernel | 23.809842 | 18.176157 | -5.633685 |
| kernel_unified_attention | 10.243747 | 0.000000 | -10.243747 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 8.698184 | 5.002237 | -3.695947 |
| mul_broadcast_2d_kernel | 5.276337 | 0.000000 | -5.276337 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 3.992079 | 2.980190 | -1.011889 |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 0.000000 | 3.982946 | +3.982946 |
| pow_func_tensor_scalar_kernel_rank_1 | 3.726596 | 0.000000 | -3.726596 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 3.629382 | 2.273878 | -1.355504 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 3.586034 | 2.943472 | -0.642562 |
| add_func_kernel_rank_1 | 3.284597 | 0.000000 | -3.284597 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 3.201777 | 2.338675 | -0.863102 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 2.784509 | 1.657556 | -1.126953 |
| mean_dim_kernel_inner | 2.673572 | 0.000000 | -2.673572 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 2.392927 | 1.804050 | -0.588876 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| _C_custom_ar::all_reduce | 1.273635 | 32.342589 | +31.068954 |
| vllm::moe_forward_shared | 26.120274 | 18.176157 | -7.944117 |
| aten::copy_ | 15.333580 | 9.102034 | -6.231546 |
| aten::mm | 12.769159 | 9.544375 | -3.224784 |
| vllm::unified_attention_with_output | 10.414276 | 0.000000 | -10.414276 |
| vllm::qwen_gdn_attention_core | 8.357334 | 6.267845 | -2.089489 |
| aten::mul | 7.287311 | 4.963714 | -2.323597 |
| aten::add | 4.209317 | 3.088885 | -1.120432 |
| aten::pow | 3.726596 | 1.514572 | -2.212023 |
| symm_mem::two_shot_all_reduce_ | 3.586034 | 2.943472 | -0.642562 |
| _vllm_fa3_C::fwd | 0.000000 | 3.043474 | +3.043474 |
| aten::mean | 2.673572 | 1.944367 | -0.729205 |
| aten::sum | 0.000000 | 1.907483 | +1.907483 |
| _C::silu_and_mul | 0.000000 | 0.976338 | +0.976338 |
| _moe_C::moe_align_block_size | 0.000000 | 0.832158 | +0.832158 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
