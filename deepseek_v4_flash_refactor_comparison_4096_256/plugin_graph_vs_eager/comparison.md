# Runtime profile comparison

- Left: `plugin_graph`
- Right: `plugin_eager`
- Percentages use total rank-0 runtime kernel duration as denominator.
- Known operator and shape counts exclude the explicit `null` metadata bucket.
- Missing logical metadata is retained in all CSV files and in mapping coverage.

## Scope summary

| Metric | Left | Right | Intersection | Left only | Right only |
|---|---:|---:|---:|---:|---:|
| CPU operator types | 112 | 110 | 106 | 6 | 4 |
| Kernel types | 144 | 155 | 136 | 8 | 19 |
| Known operator types | 43 | 41 | 38 | 5 | 3 |
| Known shape/dtype variants | 1596 | 2882 | 1275 | 321 | 1607 |

## Runtime totals

| Metric | Left | Right | Right / Left |
|---|---:|---:|---:|
| Output tokens | 16384.000000 | 16384.000000 | 1.000000 |
| Batch wall time (s) | 16.326000 | 78.676519 | 4.819093 |
| Kernel events | 669641.000000 | 689411.000000 | 1.029523 |
| Kernel duration (us) | 12908520.800000 | 11998409.944000 | 0.929495 |

## Largest kernel shares

| kernel_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 4, 8, 4, false, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 19.322077 | 20.766833 | +1.444756 |
| void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long) | 14.128169 | 10.505954 | -3.622215 |
| void sm90::fwd::sparse_attn_fwd_kernel<sm90::fwd::KernelTemplate<512, true>, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnFwdParams, sm90::fwd::KernelTemplate<512, true>::TmaParams<cute::tuple<int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 9.894761 | 10.632394 | +0.737632 |
| void marlin_moe_wna16::Marlin<1125899906909960l, 562949953487106l, 1125899906909960l, 2814749767106568l, 128, 1, 8, 4, true, 4, 2, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, int*, bool, bool, bool) | 7.565595 | 7.998681 | +0.433086 |
| mhc_post_tilelang_kernel | 5.094641 | 5.496040 | +0.401400 |
| void per_token_group_quant_8bit_kernel<c10::BFloat16, c10::Float8_e4m3fn, true, true, float>(c10::BFloat16 const*, void*, float*, int, int, int, float, float, float, int, int) | 3.985420 | 3.722738 | -0.262683 |
| mhc_pre_big_fuse_with_norm_tilelang_kernel | 3.568187 | 3.814708 | +0.246522 |
| void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > > >(SparseAttnDecodeParams, sm90::decode::sparse_fp8::KernelTemplate<(ModelType)1, 64>::TmaParams<cute::tuple<int, int, int, int>, cute::TiledCopy<cute::Copy_Atom<cute::Copy_Traits<cute::SM90_TMA_LOAD, cute::C<65536>, cute::AuxTmaParams<cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> >, cute::Layout<cute::tuple<cute::C<64>, cute::C<64>, cute::C<1>, cute::C<1> >, cute::tuple<cute::ScaledBasis<cute::C<1>, 1>, cute::ScaledBasis<cute::C<1>, 0>, cute::ScaledBasis<cute::C<1>, 2>, cute::ScaledBasis<cute::C<1>, 3> > > const&, cute::Swizzle<3, 4, 3> const&> >, cutlass::bfloat16_t>, cute::Layout<cute::tuple<cute::C<1>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<64> >, cute::C<8> > > >, cute::tuple<cute::C<0>, cute::tuple<cute::tuple<cute::tuple<cute::C<64>, cute::C<1> >, cute::C<4096> > > > >, cute::tuple<cute::C<64>, cute::C<512> > > >) | 2.865591 | 3.053981 | +0.188390 |
| void vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid<c10::BFloat16, 64>(c10::BFloat16 const*, c10::BFloat16*, c10::BFloat16 const*, unsigned char*, long const*, long const*, float const*, float, int, int, int, int, int) | 2.525740 | 2.705700 | +0.179960 |
| void deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24u, 16384u, 64u, 32u, 64u, 1u, 128u, 12u, 128u, 128u>(unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, float*) | 2.129513 | 2.290635 | +0.161122 |
| moe_sum_kernel | 1.895201 | 2.043424 | +0.148223 |
| nvjet_tss_256x128_64x4_1x2_h_bz_coopA_TNT | 1.618952 | 1.741115 | +0.122162 |
| void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 1024u, 1u, 256u, 128u, 128u, 128u, 128u, 128u, 3u, 128u, 256u, 2u, true, 132u, (deep_gemm::GemmType)0, deep_gemm::epilogue::transform::EpilogueIdentity>(float*, int*, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st) | 1.330157 | 1.422700 | +0.092543 |
| _fused_inv_rope_fp8_quant_per_head | 1.103455 | 1.126154 | +0.022699 |
| kernel_cutlass_kernel_vllmmodelsdeepseek_v4nvidiaopssparse_attn_compress_cutedslSparseAttnCompressNormRopeStoreC4Kernel_object_at__tensorptrf32gmemalign16o2048i64div16i64div161_tensorptri_0 | 1.093041 | 0.745200 | -0.347841 |

## Largest attributed operator shares

| operator_name | Left % | Right % | Delta pp |
|---|---:|---:|---:|
| _moe_C::moe_wna16_marlin_gemm | 19.322077 | 28.776185 | +9.454108 |
| symm_mem::multimem_all_reduce_ | 11.616686 | 10.505954 | -1.110732 |
| _flashmla_C::sparse_prefill_fwd | 9.894761 | 10.632394 | +0.737632 |
| vllm::dynamic_flashinfer_deepgemm_blockscale_gemm | 3.421691 | 6.340660 | +2.918969 |
| aten::mm | 2.269870 | 4.526209 | +2.256338 |
| _flashmla_C::sparse_decode_fwd | 0.366941 | 4.049528 | +3.682587 |
| _C::per_token_group_fp8_quant | 2.319047 | 3.722738 | +1.403691 |
| _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | 2.525740 | 3.122204 | +0.596464 |
| vllm::moe_forward_shared | 1.690029 | 2.043424 | +0.353395 |
| vllm::fused_inv_rope_fp8_quant_kernel | 0.859083 | 1.126154 | +0.267071 |
| aten::add | 0.734059 | 1.066418 | +0.332359 |
| vllm::sparse_attn_indexer | 0.488563 | 0.864312 | +0.375748 |
| _moe_C::moe_align_block_size | 0.273256 | 0.724338 | +0.451082 |
| _moe_C::topk_softplus_sqrt | 0.189659 | 0.627649 | +0.437990 |
| _C::silu_and_mul_with_clamp | 0.337930 | 0.556398 | +0.218468 |

See `kernel_comparison.csv`, `operator_comparison.csv`, and `shape_dtype_comparison.csv` for the complete union.
