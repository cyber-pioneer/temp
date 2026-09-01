# Runtime profile comparison

- Left: `graph_4096_512`
- Right: `eager_4096_512`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 112 | 102 | 93 | 19 | 9 |
| Kernel types | 93 | 104 | 75 | 18 | 29 |
| Known operator types | 46 | 35 | 30 | 16 | 5 |
| Known shape/dtype variants | 1334 | 3863 | 1213 | 121 | 2650 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 32768.000000 | 32768.000000 | 1.000000 |
| Batch wall time (s) | 13.685305 | 172.478111 | 12.603162 |
| Kernel events | 726240.000000 | 1676706.000000 | 2.308749 |
| Kernel duration (us) | 9228845.981000 | 12849214.170000 | 1.392288 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 35.092109 | 25.531403 | -9.560706 |
| kernel_unified_attention | 12.107101 | 8.746254 | -3.360847 |
| mm_kernel_general_host_tma | 11.540288 | 0.000000 | -11.540288 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 0.282404 | 7.901720 | +7.619315 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 7.358026 | 5.268024 | -2.090003 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 1.619253 | 6.866770 | +5.247518 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 4.477794 | 0.000000 | -4.477794 |
| mul_broadcast_2d_kernel | 0.739943 | 4.128402 | +3.388458 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 1.184055 | 3.589085 | +2.405029 |
| pow_func_tensor_scalar_kernel_rank_1 | 0.000000 | 2.901804 | +2.901804 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 0.000000 | 2.885513 | +2.885513 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 0.336580 | 2.754506 | +2.417926 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 2.649387 | 1.716301 | -0.933086 |
| add_func_kernel_rank_1 | 0.013694 | 2.467583 | +2.453890 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 2.122891 | 1.526677 | -0.596214 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 9.091416 | 28.205524 | +19.114108 |
| aten::copy_ | 0.707350 | 13.762388 | +13.055039 |
| aten::mm | 8.193317 | 11.793481 | +3.600164 |
| vllm::unified_attention_with_output | 6.318788 | 8.968080 | +2.649293 |
| vllm::qwen_gdn_attention_core | 3.567176 | 8.258811 | +4.691636 |
| symm_mem::two_shot_all_reduce_ | 1.619253 | 6.866770 | +5.247518 |
| aten::mul | 0.302156 | 5.813419 | +5.511263 |
| _C_custom_ar::all_reduce | 0.038031 | 3.589085 | +3.551054 |
| aten::add | 0.013694 | 3.569110 | +3.555416 |
| aten::pow | 0.000000 | 2.901804 | +2.901804 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 2.893377 | 0.000000 | -2.893377 |
| aten::mean | 0.000000 | 2.112189 | +2.112189 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.869763 | 0.000000 | -0.869763 |
| aten::rsqrt | 0.000000 | 0.602205 | +0.602205 |
| aten::zeros | 0.027805 | 0.531643 | +0.503838 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
