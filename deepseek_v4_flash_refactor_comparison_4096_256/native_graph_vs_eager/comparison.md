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
| Batch wall time (s) | 15.603411 | 53.257877 | 3.413220 |
| Kernel events | 657290.000000 | 602756.000000 | 0.917032 |
| Kernel duration (us) | 12956103.829000 | 29810023.945000 | 2.300848 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long) | 13.690151 | 59.852465 | +46.162314 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 4, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 19.116662 | 8.346023 | -10.770639 |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnFwdParams, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 9.807672 | 4.287781 | -5.519891 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 7.609736 | 3.226275 | -4.383462 |
| mhc_post_tilelang_kernel | 5.077080 | 2.211195 | -2.865884 |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 const*, void*, float*, int, int, int, float, float, float, int, int) | 4.017405 | 1.547365 | -2.470040 |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 3.550397 | 1.540469 | -2.009928 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 0.058840 | 3.496545 | +3.437705 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 2.995938 | 1.294579 | -1.701359 |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnDecodeParams, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 2.851776 | 1.236484 | -1.615291 |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 64>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int) | 2.522616 | 1.096428 | -1.426188 |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*) | 2.121009 | 0.924936 | -1.196072 |
| nvjet_tss_256x128_64x4_1x2_h_bz_coopA_TNT | 1.607329 | 0.699838 | -0.907491 |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 1024u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 2u, true, 132u, (deep_gemm::GemmType)0, deep_gemm::epilogue::transform::EpilogueIdentity>(float*, int*, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st) | 1.323661 | 0.576466 | -0.747195 |
| _fused_inv_rope_fp8_quant_per_head | 1.101621 | 0.463535 | -0.638086 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| symm_mem::multimem_all_reduce_ | 11.441448 | 59.852465 | +48.411016 |
| _moe_C::moe_wna16_marlin_gemm | 19.116662 | 11.576681 | -7.539981 |
| _flashmla_C::sparse_prefill_fwd | 9.807672 | 4.287781 | -5.519891 |
| _C_custom_ar::all_reduce | 0.000000 | 3.496545 | +3.496545 |
| vllm::dynamic_flashinfer_deepgemm_blockscale_gemm | 3.406743 | 2.666681 | -0.740062 |
| _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | 2.522616 | 1.275356 | -1.247259 |
| aten::sum | 2.481433 | 1.294579 | -1.186854 |
| _C::per_token_group_fp8_quant | 2.316436 | 1.547365 | -0.769071 |
| aten::mm | 2.254500 | 1.942621 | -0.311879 |
| _flashmla_C::sparse_decode_fwd | 0.364244 | 1.633040 | +1.268796 |
| vllm::fused_inv_rope_fp8_quant_kernel | 0.855992 | 0.463535 | -0.392457 |
| aten::add | 0.732895 | 0.377355 | -0.355541 |
| vllm::sparse_attn_indexer | 0.487920 | 0.351547 | -0.136373 |
| _C::top_k_per_row_prefill | 0.348935 | 0.150914 | -0.198021 |
| _C::silu_and_mul_with_clamp | 0.337619 | 0.324096 | -0.013523 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
