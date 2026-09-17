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
| Kernel types | 104 | 118 | 57 | 47 | 61 |
| Known operator types | 35 | 36 | 24 | 11 | 12 |
| Known shape/dtype variants | 3863 | 3212 | 960 | 2903 | 2252 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 95.814762 | 32.210468 | 0.336174 |
| Kernel events | 896162.000000 | 707105.000000 | 0.789037 |
| Kernel duration (us) | 8838866.189000 | 7640940.599000 | 0.864471 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 22.391059 | 25.989950 | +3.598891 |
| kernel_unified_attention | 9.502757 | 0.000000 | -9.502757 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 8.086748 | 7.185019 | -0.901729 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 7.449504 | 3.835734 | -3.613770 |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 0.000000 | 5.732383 | +5.732383 |
| mul_broadcast_2d_kernel | 4.901841 | 0.000000 | -4.901841 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 0.000000 | 4.291536 | +4.291536 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 3.875668 | 3.713796 | -0.161872 |
| _fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta | 3.700809 | 0.000000 | -3.700809 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 2.973322 | 3.471058 | +0.497736 |
| pow_func_tensor_scalar_kernel_rank_1 | 3.460861 | 0.000000 | -3.460861 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 3.368196 | 3.279961 | -0.088236 |
| void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > > >(flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<128>, cute::C<80>, cute::C<256> >, 256, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<128>, cute::C<256>, cute::C<80> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 256, true, true, true, false, 1>, flash::VarlenDynamicPersistentTileScheduler<128, 80, 256, 128, true, true, true, false, false, true> > >::Params) | 0.000000 | 3.259918 | +3.259918 |
| add_func_kernel_rank_1 | 3.059013 | 0.000000 | -3.059013 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4, 4>) | 0.000000 | 2.794770 | +2.794770 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 24.534336 | 25.989950 | +1.455614 |
| aten::copy_ | 14.250476 | 13.089267 | -1.161208 |
| aten::mm | 11.880798 | 13.662943 | +1.782145 |
| vllm::unified_attention_with_output | 9.661287 | 0.000000 | -9.661287 |
| vllm::qwen_gdn_attention_core | 7.749021 | 8.940793 | +1.191772 |
| symm_mem::two_shot_all_reduce_ | 7.449504 | 3.835734 | -3.613770 |
| aten::mul | 6.769724 | 7.101000 | +0.331276 |
| aten::add | 3.918123 | 4.435841 | +0.517718 |
| _vllm_fa3_C::fwd | 0.000000 | 4.382237 | +4.382237 |
| _C_custom_ar::all_reduce | 3.875668 | 3.713796 | -0.161872 |
| aten::pow | 3.460861 | 2.179134 | -1.281727 |
| aten::mean | 2.484565 | 2.794770 | +0.310205 |
| aten::sum | 0.000000 | 2.734705 | +2.734705 |
| _C::silu_and_mul | 0.000000 | 1.396865 | +1.396865 |
| _moe_C::moe_align_block_size | 0.000000 | 1.199187 | +1.199187 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
