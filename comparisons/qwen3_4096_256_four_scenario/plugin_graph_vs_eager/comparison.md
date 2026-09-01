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
| Kernel types | 96 | 104 | 72 | 24 | 32 |
| Known operator types | 46 | 35 | 30 | 16 | 5 |
| Known shape/dtype variants | 1384 | 3863 | 779 | 605 | 3084 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 16.763174 | 93.152467 | 5.556971 |
| Kernel events | 431551.000000 | 896162.000000 | 2.076607 |
| Kernel duration (us) | 9715739.225000 | 8207711.519000 | 0.844785 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| mm_kernel_general_host_tma | 44.102663 | 0.000000 | -44.102663 |
| fused_moe_kernel | 20.538074 | 23.809842 | +3.271768 |
| kernel_unified_attention | 8.633601 | 10.243747 | +1.610147 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 0.133943 | 8.698184 | +8.564241 |
| mul_broadcast_2d_kernel | 0.498666 | 5.276337 | +4.777671 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 3.384799 | 3.992079 | +0.607280 |
| void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)1, __nv_bfloat16, 2, true, true>(flashinfer::trtllm_allreduce_fusion::AllReduceFusionParams<__nv_bfloat16>) | 3.946131 | 0.000000 | -3.946131 |
| pow_func_tensor_scalar_kernel_rank_1 | 0.000000 | 3.726596 | +3.726596 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 0.000000 | 3.629382 | +3.629382 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 1.555652 | 3.586034 | +2.030382 |
| add_func_kernel_rank_1 | 0.007251 | 3.284597 | +3.277346 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 0.000000 | 3.201777 | +3.201777 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 0.210601 | 2.784509 | +2.573908 |
| mean_dim_kernel_inner | 0.000000 | 2.673572 | +2.673572 |
| void cutlass::device_kernel<flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > > >(flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule<flat::collective::FlatMainloopTmaWarpSpecializedDeltaRule<cutlass::bfloat16_t, float, float, cute::tuple<cute::C<64>, cute::C<64>, cute::C<128> >, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, cute::tuple<long, cute::C<1>, int>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >, flat::kernel::IndividualTileScheduler<flat::kernel::GVATag>, std::tuple<flat::kernel::Option<(flat::kernel::Tag)1, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)11, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)10, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)9, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)12, cute::C<true> >, flat::kernel::Option<(flat::kernel::Tag)13, cute::C<false> > > >::Params) | 2.036003 | 2.392927 | +0.356923 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| aten::mm | 43.046178 | 12.769159 | -30.277019 |
| vllm::moe_forward_shared | 8.723336 | 26.120274 | +17.396938 |
| aten::copy_ | 0.413490 | 15.333580 | +14.920089 |
| vllm::unified_attention_with_output | 6.014236 | 10.414276 | +4.400041 |
| vllm::qwen_gdn_attention_core | 3.417217 | 8.357334 | +4.940117 |
| aten::mul | 0.288345 | 7.287311 | +6.998966 |
| aten::add | 0.007251 | 4.209317 | +4.202066 |
| aten::pow | 0.000000 | 3.726596 | +3.726596 |
| symm_mem::two_shot_all_reduce_ | 1.555652 | 3.586034 | +2.030382 |
| vllm::flashinfer_trtllm_fused_allreduce_norm | 3.171772 | 0.000000 | -3.171772 |
| aten::mean | 0.000000 | 2.673572 | +2.673572 |
| _C_custom_ar::all_reduce | 0.023214 | 1.273635 | +1.250421 |
| triton_red_fused__to_copy_add_copy__fused_add_rms_norm_moe_forward_shared_3 | 0.827102 | 0.000000 | -0.827102 |
| aten::cat | 0.310973 | 0.665210 | +0.354237 |
| aten::silu | 0.000000 | 0.662319 | +0.662319 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
