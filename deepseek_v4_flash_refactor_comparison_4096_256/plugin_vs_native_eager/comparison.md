# Runtime profile comparison

- Left: `plugin_eager`
- Right: `native_eager`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 110 | 109 | 103 | 7 | 6 |
| Kernel types | 155 | 155 | 132 | 23 | 23 |
| Known operator types | 41 | 36 | 31 | 10 | 5 |
| Known shape/dtype variants | 2882 | 2759 | 2097 | 785 | 662 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 78.676519 | 53.257877 | 0.676922 |
| Kernel events | 689411.000000 | 602756.000000 | 0.874306 |
| Kernel duration (us) | 11998409.944000 | 29810023.945000 | 2.484498 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long) | 10.505954 | 59.852465 | +49.346511 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 4, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 20.766833 | 8.346023 | -12.420810 |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnFwdParams, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 10.632394 | 4.287781 | -6.344612 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 7.998681 | 3.226275 | -4.772407 |
| mhc_post_tilelang_kernel | 5.496040 | 2.211195 | -3.284845 |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 3.814708 | 1.540469 | -2.274240 |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 const*, void*, float*, int, int, int, float, float, float, int, int) | 3.722738 | 1.547365 | -2.175372 |
| void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData*, vllm::RankSignals, vllm::Signal*, __nv_bfloat16*, int, int) | 0.201353 | 3.496545 | +3.295192 |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnDecodeParams, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 3.053981 | 1.236484 | -1.817497 |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 64>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int) | 2.705700 | 1.096428 | -1.609272 |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*) | 2.290635 | 0.924936 | -1.365699 |
| moe_sum_kernel | 2.043424 | 0.000000 | -2.043424 |
| nvjet_tss_256x128_64x4_1x2_h_bz_coopA_TNT | 1.741115 | 0.699838 | -1.041277 |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 1024u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 2u, true, 132u, (deep_gemm::GemmType)0, deep_gemm::epilogue::transform::EpilogueIdentity>(float*, int*, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st) | 1.422700 | 0.576466 | -0.846234 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 0.000000 | 1.294579 | +1.294579 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| symm_mem::multimem_all_reduce_ | 10.505954 | 59.852465 | +49.346511 |
| _moe_C::moe_wna16_marlin_gemm | 28.776185 | 11.576681 | -17.199505 |
| _flashmla_C::sparse_prefill_fwd | 10.632394 | 4.287781 | -6.344612 |
| vllm::dynamic_flashinfer_deepgemm_blockscale_gemm | 6.340660 | 2.666681 | -3.673979 |
| aten::mm | 4.526209 | 1.942621 | -2.583588 |
| _flashmla_C::sparse_decode_fwd | 4.049528 | 1.633040 | -2.416488 |
| _C::per_token_group_fp8_quant | 3.722738 | 1.547365 | -2.175372 |
| _C_custom_ar::all_reduce | 0.201353 | 3.496545 | +3.295192 |
| _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | 3.122204 | 1.275356 | -1.846848 |
| vllm::moe_forward_shared | 2.043424 | 0.000000 | -2.043424 |
| aten::sum | 0.000000 | 1.294579 | +1.294579 |
| vllm::fused_inv_rope_fp8_quant_kernel | 1.126154 | 0.463535 | -0.662619 |
| aten::add | 1.066418 | 0.377355 | -0.689063 |
| vllm::sparse_attn_indexer | 0.864312 | 0.351547 | -0.512764 |
| _moe_C::moe_align_block_size | 0.724338 | 0.303018 | -0.421320 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
