# Runtime profile comparison

- Left: `4096_512`
- Right: `4096_256`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 102 | 102 | 102 | 0 | 0 |
| Kernel types | 104 | 104 | 104 | 0 | 0 |
| Known operator types | 35 | 35 | 35 | 0 | 0 |
| Known shape/dtype variants | 3863 | 3863 | 3863 | 0 | 0 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 32768.000000 | 16384.000000 | 0.500000 |
| Batch wall time (s) | 172.478111 | 94.719346 | 0.549167 |
| Kernel events | 1676706.000000 | 896162.000000 | 0.534478 |
| Kernel duration (us) | 12849214.170000 | 8241954.500000 | 0.641436 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 25.531403 | 24.118347 | -1.413056 |
| kernel_unified_attention | 8.746254 | 10.193581 | +1.447327 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 7.901720 | 8.667631 | +0.765911 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 6.866770 | 3.569709 | -3.297061 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 5.268024 | 3.978266 | -1.289757 |
| mul_broadcast_2d_kernel | 4.128402 | 5.255386 | +1.126984 |
| pow_func_tensor_scalar_kernel_rank_1 | 2.901804 | 3.711506 | +0.809701 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 2.885513 | 3.615964 | +0.730451 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 3.589085 | 1.263280 | -2.325804 |
| add_func_kernel_rank_1 | 2.467583 | 3.271128 | +0.803545 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 2.044388 | 3.185936 | +1.141547 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 2.754506 | 2.775176 | +0.020670 |
| mean_dim_kernel_inner | 2.112189 | 2.662475 | +0.550286 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 1.526677 | 2.378674 | +0.851997 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 1.392368 | 2.171109 | +0.778742 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 28.205524 | 26.422647 | -1.782877 |
| aten::copy_ | 13.762388 | 15.279406 | +1.517018 |
| aten::mm | 11.793481 | 12.721200 | +0.927719 |
| vllm::unified_attention_with_output | 8.968080 | 10.363550 | +1.395470 |
| vllm::qwen_gdn_attention_core | 8.258811 | 8.321305 | +0.062494 |
| aten::mul | 5.813419 | 7.258438 | +1.445019 |
| symm_mem::two_shot_all_reduce_ | 6.866770 | 3.569709 | -3.297061 |
| aten::add | 3.569110 | 4.192459 | +0.623349 |
| aten::pow | 2.901804 | 3.711506 | +0.809701 |
| _C_custom_ar::all_reduce | 3.589085 | 1.263280 | -2.325804 |
| aten::mean | 2.112189 | 2.662475 | +0.550286 |
| aten::cat | 0.527632 | 0.662366 | +0.134734 |
| aten::silu | 0.503682 | 0.659681 | +0.155999 |
| aten::rsqrt | 0.602205 | 0.504502 | -0.097703 |
| aten::zeros | 0.531643 | 0.491350 | -0.040293 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
