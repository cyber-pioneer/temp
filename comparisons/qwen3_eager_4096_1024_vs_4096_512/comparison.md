# Runtime profile comparison

- Left: `4096_1024`
- Right: `4096_512`
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
| Output tokens | 65536.000000 | 32768.000000 | 0.500000 |
| Batch wall time (s) | 355.902954 | 172.478111 | 0.484621 |
| Kernel events | 3355706.000000 | 1676706.000000 | 0.499658 |
| Kernel duration (us) | 20888691.049000 | 12849214.170000 | 0.615128 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| fused_moe_kernel | 28.992602 | 25.531403 | -3.461199 |
| kernel_unified_attention | 8.307939 | 8.746254 | +0.438315 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>) | 7.705604 | 7.901720 | +0.196116 |
| void (anonymous namespace)::two_shot_all_reduce_kernel_inplace<c10::BFloat16, 16, 2>(c10::BFloat16**, unsigned long, unsigned long, unsigned int**, unsigned long, unsigned long) | 1.559846 | 6.866770 | +5.306924 |
| fused_recurrent_gated_delta_rule_packed_decode_kernel | 6.577160 | 5.268024 | -1.309136 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 6.347077 | 3.589085 | -2.757993 |
| mul_broadcast_2d_kernel | 3.951677 | 4.128402 | +0.176724 |
| pow_func_tensor_scalar_kernel_rank_1 | 2.425172 | 2.901804 | +0.476633 |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}) | 2.887100 | 2.754506 | -0.132594 |
| void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>) | 2.474876 | 2.885513 | +0.410637 |
| add_func_kernel_rank_1 | 1.961090 | 2.467583 | +0.506493 |
| nvjet_tst_64x8_64x16_2x4_h_bz_TNT | 2.131832 | 1.716301 | -0.415532 |
| mean_dim_kernel_inner | 1.789302 | 2.112189 | +0.322887 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 1.251215 | 2.044388 | +0.793173 |
| nvjet_tst_64x32_64x16_4x2_h_bz_TNT | 1.876429 | 1.472264 | -0.404164 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| vllm::moe_forward_shared | 32.080800 | 28.205524 | -3.875276 |
| aten::copy_ | 13.303076 | 13.762388 | +0.459312 |
| aten::mm | 11.734555 | 11.793481 | +0.058926 |
| vllm::unified_attention_with_output | 8.581348 | 8.968080 | +0.386733 |
| vllm::qwen_gdn_attention_core | 8.685793 | 8.258811 | -0.426981 |
| symm_mem::two_shot_all_reduce_ | 1.559846 | 6.866770 | +5.306924 |
| _C_custom_ar::all_reduce | 6.347029 | 3.589085 | -2.757944 |
| aten::mul | 5.873137 | 5.813419 | -0.059718 |
| aten::add | 3.240359 | 3.569110 | +0.328751 |
| aten::pow | 2.425143 | 2.901804 | +0.476661 |
| aten::mean | 1.789289 | 2.112189 | +0.322900 |
| aten::rsqrt | 0.718076 | 0.602205 | -0.115872 |
| aten::zeros | 0.596438 | 0.531643 | -0.064795 |
| aten::cat | 0.452866 | 0.527632 | +0.074766 |
| aten::silu | 0.412397 | 0.503682 | +0.091285 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
