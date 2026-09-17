# Runtime profile comparison

- Left: `plugin_graph`
- Right: `plugin_eager`
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
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 10.542003 | 95.814762 | 9.088858 |
| Kernel events | 390624.000000 | 896162.000000 | 2.294181 |
| Kernel duration (us) | 6125271.191000 | 8838866.189000 | 1.443016 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 31.833558 | 22.391059 | -9.442500 |
| kernel_unified_attention | 13.680540 | 9.502757 | -4.177783 |
| mm_kernel_general_host_tma | 13.159865 | 0.000000 | -13.159865 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 0.212129 | 8.086748 | +7.874620 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 2.435140 | 7.449504 | +5.014363 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 5.542854 | 0.000000 | -5.542854 |
| _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta | 5.358402 | 3.700809 | -1.657594 |
| mul_broadcast_2d_kernel | 0.792116 | 4.901841 | +4.109725 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 0.902095 | 3.875668 | +2.973573 |
| pow_func_tensor_scalar_kernel_rank_1 | 0.000000 | 3.460861 | +3.460861 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 0.000000 | 3.368196 | +3.368196 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 3.204148 | 2.218525 | -0.985624 |
| add_func_kernel_rank_1 | 0.011133 | 3.059013 | +3.047880 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 0.000000 | 2.973322 | +2.973322 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 0.329396 | 2.589493 | +2.260097 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 13.707147 | 24.534336 | +10.827189 |
| aten::copy_ | 0.648419 | 14.250476 | +13.602057 |
| aten::mm | 11.565523 | 11.880798 | +0.315275 |
| vllm::unified_attention_with_output | 9.542867 | 9.661287 | +0.118420 |
| vllm::qwen_gdn_attention_core | 5.379497 | 7.749021 | +2.369524 |
| symm_mem::two_shot_all_reduce_ | 2.435140 | 7.449504 | +5.014363 |
| aten::mul | 0.460983 | 6.769724 | +6.308741 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 4.358472 | 0.000000 | -4.358472 |
| aten::add | 0.011133 | 3.918123 | +3.906990 |
| _C_custom_ar::all_reduce | 0.025202 | 3.875668 | +3.850466 |
| aten::pow | 0.000000 | 3.460861 | +3.460861 |
| aten::mean | 0.000000 | 2.484565 | +2.484565 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 1.312816 | 0.000000 | -1.312816 |
| aten::cat | 0.493488 | 0.620358 | +0.126870 |
| aten::silu | 0.000000 | 0.615380 | +0.615380 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
