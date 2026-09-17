# Runtime profile comparison

- Left: `plugin_graph`
- Right: `native_graph`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 112 | 114 | 109 | 3 | 5 |
| Kernel types | 144 | 151 | 122 | 22 | 29 |
| Known operator types | 43 | 38 | 35 | 8 | 3 |
| Known shape/dtype variants | 1596 | 1675 | 1359 | 237 | 316 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 16.326000 | 15.603411 | 0.955740 |
| Kernel events | 669641.000000 | 657290.000000 | 0.981556 |
| Kernel duration (us) | 12908520.800000 | 12956103.829000 | 1.003686 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 4, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 19.322077 | 19.116662 | -0.205416 |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long) | 14.128169 | 13.690151 | -0.438019 |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnFwdParams, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 9.894761 | 9.807672 | -0.087089 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 7.565595 | 7.609736 | +0.044141 |
| mhc_post_tilelang_kernel | 5.094641 | 5.077080 | -0.017561 |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 const*, void*, float*, int, int, int, float, float, float, int, int) | 3.985420 | 4.017405 | +0.031985 |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 3.568187 | 3.550397 | -0.017790 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8> >(at::native::ReduceOp<c10::BFloat16, at::native::func_wrapper_t<c10::BFloat16, at::native::sum_functor<c10::BFloat16, float, c10::BFloat16>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, c10::BFloat16, 4, 8>) | 0.000000 | 2.995938 | +2.995938 |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnDecodeParams, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 2.865591 | 2.851776 | -0.013816 |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 64>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int) | 2.525740 | 2.522616 | -0.003124 |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*) | 2.129513 | 2.121009 | -0.008504 |
| moe_sum_kernel | 1.895201 | 0.000000 | -1.895201 |
| nvjet_tss_256x128_64x4_1x2_h_bz_coopA_TNT | 1.618952 | 1.607329 | -0.011623 |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 1024u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 2u, true, 132u, (deep_gemm::GemmType)0, deep_gemm::epilogue::transform::EpilogueIdentity>(float*, int*, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st) | 1.330157 | 1.323661 | -0.006495 |
| _fused_inv_rope_fp8_quant_per_head | 1.103455 | 1.101621 | -0.001835 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| _moe_C::moe_wna16_marlin_gemm | 19.322077 | 19.116662 | -0.205416 |
| symm_mem::multimem_all_reduce_ | 11.616686 | 11.441448 | -0.175237 |
| _flashmla_C::sparse_prefill_fwd | 9.894761 | 9.807672 | -0.087089 |
| vllm::dynamic_flashinfer_deepgemm_blockscale_gemm | 3.421691 | 3.406743 | -0.014948 |
| _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | 2.525740 | 2.522616 | -0.003124 |
| aten::sum | 0.000000 | 2.481433 | +2.481433 |
| _C::per_token_group_fp8_quant | 2.319047 | 2.316436 | -0.002611 |
| aten::mm | 2.269870 | 2.254500 | -0.015371 |
| vllm::moe_forward_shared | 1.690029 | 0.000000 | -1.690029 |
| vllm::fused_inv_rope_fp8_quant_kernel | 0.859083 | 0.855992 | -0.003091 |
| aten::add | 0.734059 | 0.732895 | -0.001163 |
| vllm::sparse_attn_indexer | 0.488563 | 0.487920 | -0.000643 |
| aten::repeat | 0.380007 | 0.000000 | -0.380007 |
| _flashmla_C::sparse_decode_fwd | 0.366941 | 0.364244 | -0.002696 |
| _C::top_k_per_row_prefill | 0.349718 | 0.348935 | -0.000783 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
