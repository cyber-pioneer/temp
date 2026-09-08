# Runtime profile comparison

- Left: `native_graph`
- Right: `native_eager`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 114 | 109 | 106 | 8 | 3 |
| Kernel types | 151 | 155 | 140 | 11 | 15 |
| Known operator types | 38 | 36 | 33 | 5 | 3 |
| Known shape/dtype variants | 1675 | 2759 | 1370 | 305 | 1389 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 15.469345 | 53.292541 | 3.445042 |
| Kernel events | 657290.000000 | 602756.000000 | 0.917032 |
| Kernel duration (us) | 12329047.725000 | 30737180.086000 | 2.493070 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long) | 9.345302 | 61.168990 | +51.823689 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 4, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 20.077531 | 8.082545 | -11.994986 |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnFwdParams, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 10.302691 | 4.155570 | -6.147121 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 8.000071 | 3.133554 | -4.866517 |
| mhc_post_tilelang_kernel | 5.340679 | 2.145732 | -3.194947 |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 const*, void*, float*, int, int, int, float, float, float, int, int) | 4.220127 | 1.506048 | -2.714078 |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 3.731114 | 1.493924 | -2.237190 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 0.059364 | 3.282520 | +3.223156 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 3.147809 | 1.255901 | -1.891908 |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnDecodeParams, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 2.996698 | 1.199524 | -1.797175 |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 64>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int) | 2.651860 | 1.064075 | -1.587785 |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*) | 2.227486 | 0.896707 | -1.330778 |
| nvjet_tss_256x128_64x4_1x2_h_bz_coopA_TNT | 1.689289 | 0.678198 | -1.011090 |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 1024u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 2u, true, 132u, (deep_gemm::GemmType)0, deep_gemm::epilogue::transform::EpilogueIdentity>(float*, int*, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st) | 1.391151 | 0.557870 | -0.833281 |
| _fused_inv_rope_fp8_quant_per_head | 1.157963 | 0.450277 | -0.707686 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| symm_mem::multimem_all_reduce_ | 6.443962 | 61.168990 | +54.725028 |
| _moe_C::moe_wna16_marlin_gemm | 20.077531 | 11.220315 | -8.857217 |
| _flashmla_C::sparse_prefill_fwd | 10.302691 | 4.155570 | -6.147121 |
| vllm::dynamic_flashinfer_deepgemm_blockscale_gemm | 3.577170 | 2.593760 | -0.983410 |
| _C_custom_ar::all_reduce | 0.000000 | 3.282520 | +3.282520 |
| _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | 2.651860 | 1.238776 | -1.413084 |
| aten::sum | 2.607157 | 1.255901 | -1.351256 |
| _C::per_token_group_fp8_quant | 2.432166 | 1.506048 | -0.926118 |
| aten::mm | 2.369186 | 1.884505 | -0.484681 |
| _flashmla_C::sparse_decode_fwd | 0.382806 | 1.583509 | +1.200703 |
| vllm::fused_inv_rope_fp8_quant_kernel | 0.899609 | 0.450277 | -0.449331 |
| aten::add | 0.770171 | 0.365976 | -0.404195 |
| vllm::sparse_attn_indexer | 0.512711 | 0.341327 | -0.171384 |
| _C::top_k_per_row_prefill | 0.366334 | 0.146335 | -0.219999 |
| _C::silu_and_mul_with_clamp | 0.354834 | 0.316036 | -0.038799 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
