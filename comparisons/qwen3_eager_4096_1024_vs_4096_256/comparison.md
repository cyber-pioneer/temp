# Runtime profile comparison

- Left: `4096_1024`
- Right: `4096_256`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 102 | 102 | 102 | 0 | 0 |
| Kernel types | 110 | 104 | 100 | 10 | 4 |
| Known operator types | 35 | 35 | 35 | 0 | 0 |
| Known shape/dtype variants | 3954 | 3863 | 1215 | 2739 | 2648 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 65536.000000 | 16384.000000 | 0.250000 |
| Batch wall time (s) | 355.902954 | 94.719346 | 0.266138 |
| Kernel events | 3355706.000000 | 896162.000000 | 0.267056 |
| Kernel duration (us) | 20888691.049000 | 8241954.500000 | 0.394565 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 28.992602 | 24.118347 | -4.874255 |
| kernel_unified_attention | 8.307939 | 10.193581 | +1.885642 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 7.705604 | 8.667631 | +0.962027 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 6.577160 | 3.978266 | -2.598894 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 6.347077 | 1.263280 | -5.083797 |
| mul_broadcast_2d_kernel | 3.951677 | 5.255386 | +1.303708 |
| pow_func_tensor_scalar_kernel_rank_1 | 2.425172 | 3.711506 | +1.286334 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 2.474876 | 3.615964 | +1.141088 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 1.559846 | 3.569709 | +2.009863 |
| add_func_kernel_rank_1 | 1.961090 | 3.271128 | +1.310038 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 1.251215 | 3.185936 | +1.934721 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 2.887100 | 2.775176 | -0.111924 |
| mean_dim_kernel_inner | 1.789302 | 2.662475 | +0.873173 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 0.949229 | 2.378674 | +1.429445 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 0.857501 | 2.171109 | +1.313608 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 32.080800 | 26.422647 | -5.658153 |
| aten::copy_ | 13.303076 | 15.279406 | +1.976330 |
| aten::mm | 11.734555 | 12.721200 | +0.986645 |
| vllm::unified_attention_with_output | 8.581348 | 10.363550 | +1.782202 |
| vllm::qwen_gdn_attention_core | 8.685793 | 8.321305 | -0.364488 |
| aten::mul | 5.873137 | 7.258438 | +1.385301 |
| _C_custom_ar::all_reduce | 6.347029 | 1.263280 | -5.083748 |
| aten::add | 3.240359 | 4.192459 | +0.952100 |
| aten::pow | 2.425143 | 3.711506 | +1.286362 |
| symm_mem::two_shot_all_reduce_ | 1.559846 | 3.569709 | +2.009863 |
| aten::mean | 1.789289 | 2.662475 | +0.873187 |
| aten::rsqrt | 0.718076 | 0.504502 | -0.213575 |
| aten::cat | 0.452866 | 0.662366 | +0.209500 |
| aten::silu | 0.412397 | 0.659681 | +0.247283 |
| aten::zeros | 0.596438 | 0.491350 | -0.105088 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
