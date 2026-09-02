# Qwen3.6-35B-A3B 4096/256 four-scenario comparison

## Scope

All four runs used the same model, tensor parallel size 2, 64 concurrent
requests, 4096 input tokens per request, and 256 output tokens per request. A
complete 64-request warmup batch finished before `/start_profile`. Only the
second batch is included in each runtime trace. Extraction uses rank 0 and
excludes CUDA Graph construction.

The native runs set `VLLM_PLUGINS=""`. The API-server process environment
contained the explicit empty value and neither native log contained
`Loading plugin fl` or `Platform plugin fl is activated`. Both plugin runs
did contain the activation message.

## Scenario summary

| Scenario | Runtime | Batch wall time (s) | Kernel events | Kernel types | Kernel duration (us) | CPU op types | Normalized attributed op types | Shape/dtype variants | Op-kernel relations | Mapping event coverage | Mapping time coverage | Rank-0 trace (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | plugin-FL, CUDA Graph | 16.763 | 431,551 | 96 | 9,715,739.225 | 112 | 44 | 1,384 | 129 | 22.218% | 70.637% | 33.487 |
| plugin_eager | plugin-FL, eager | 93.152 | 896,162 | 104 | 8,207,711.519 | 102 | 33 | 3,863 | 105 | 99.871% | 99.960% | 191.187 |
| native_graph | native vLLM, CUDA Graph | 7.733 | 347,278 | 114 | 6,522,629.903 | 117 | 47 | 1,457 | 144 | 16.962% | 56.158% | 30.269 |
| native_eager | native vLLM, eager | 32.799 | 716,851 | 123 | 11,016,806.643 | 108 | 34 | 3,349 | 125 | 98.192% | 99.428% | 163.132 |

Batch wall time is measured while profiling is active. It is useful for this
controlled comparison but is not an unprofiled throughput benchmark.

Every boolean conservation check in all four `summary.json` files is
`true`. For every run:

- trace, compact summary, and detailed report preserve the same kernel event
  count and duration
- every normalized `(operator_name, operator_kind, kernel_name)` relation is
  unique
- every physical kernel remains present in `operator_list.csv`
- every row has a positive integer ID
- one classified operator uses one stable ID across all related kernels
- unattributed NVJet kernels use one shared ID per run

Operator-list relation counts by kind:

| Scenario | aten | custom | runtime_operator | torch_compile | triton_compiled | unattributed | unattributed_nvjet |
|---|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 48 | 16 | 1 | 3 | 15 | 39 | 7 |
| plugin_eager | 79 | 19 | 3 | 3 | 0 | 1 | 0 |
| native_graph | 59 | 17 | 1 | 3 | 15 | 36 | 13 |
| native_eager | 95 | 22 | 1 | 3 | 0 | 4 | 0 |

The three `torch_compile` relations in each run share one ID and map to
`vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask`.

## Type and shape differences

Each cell below is `left / right / intersection / left-only / right-only`.

| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |
|---|---|---|---|---|
| plugin graph vs native graph | 96 / 114 / 58 / 38 / 56 | 46 / 49 / 36 / 10 / 13 | 1,384 / 1,457 / 778 / 606 / 679 | 112 / 117 / 104 / 8 / 13 |
| plugin eager vs native eager | 104 / 123 / 61 / 43 / 62 | 35 / 36 / 24 / 11 / 12 | 3,863 / 3,349 / 1,025 / 2,838 / 2,324 | 102 / 108 / 96 / 6 / 12 |
| plugin graph vs plugin eager | 96 / 104 / 72 / 24 / 32 | 46 / 35 / 30 / 16 / 5 | 1,384 / 3,863 / 779 / 605 / 3,084 | 112 / 102 / 93 / 19 / 9 |
| native graph vs native eager | 114 / 123 / 90 / 24 / 33 | 49 / 36 / 33 / 16 / 3 | 1,457 / 3,349 / 820 / 637 / 2,529 | 117 / 108 / 99 / 18 / 9 |

Raw attributed labels come from `kernel_details_report.csv`. They are kept
unchanged for trace fidelity. The normalized attributed counts in the scenario
table come from `operator_list.csv`, where three generated Triton names are
represented by their one known `torch.compile` source function.

The eager runs expose many more logical shape/dtype variants because Python and
PyTorch operators execute in the runtime profiling window. CUDA Graph replay
executes graph-internal kernels without replaying most original CPU operators.
Consequently, graph mode has complete physical kernel accounting but much lower
runtime logical-shape attribution. The graph and eager operator lists must not
be treated as interchangeable inventories.

Plugin-FL also changes the physical program rather than merely renaming
kernels. This is visible from the low plugin/native kernel intersections:
58 kernel names in graph mode and 61 in eager mode. The full union, including
one-sided entries, is preserved in the generated comparison CSV files.

## Largest kernel-share changes

Percentages use total rank-0 runtime kernel duration in the corresponding run.

### Plugin graph vs native graph

- `mm_kernel_general_host_tma`: 44.103% in plugin graph and absent in native
  graph.
- `fused_moe_kernel`: 20.538% in plugin graph versus 31.006% in native graph.
- `kernel_unified_attention`: 8.634% in plugin graph and absent in native
  graph.
- The attributed `aten::mm` share is 43.046% versus 9.643%.
- Unattributed kernel time is 29.363% versus 43.842%.

### Plugin eager vs native eager

- `vllm::cross_device_reduce_1stage`: 1.274% in plugin eager versus 32.343%
  in native eager.
- `kernel_unified_attention`: 10.244% in plugin eager and absent in native
  eager.
- `fused_moe_kernel`: 23.810% in plugin eager versus 18.176% in native eager.
- Attributed `vllm::moe_forward_shared`: 26.120% versus 18.176%.
- Attributed `aten::copy_`: 15.334% versus 9.102%.

### Graph vs eager

For plugin-FL, `mm_kernel_general_host_tma` contributes 44.103% in graph and
is absent in eager. The attributed `aten::mm` share changes from 43.046% to
12.769%, while `vllm::moe_forward_shared` changes from 8.723% to 26.120%.
Unattributed time falls from 29.363% to 0.040%.

For native vLLM, `vllm::cross_device_reduce_1stage` changes from 1.007% in
graph to 32.343% in eager, while `fused_moe_kernel` changes from 31.006% to
18.176%. Unattributed time falls from 43.842% to 0.572%.

## Result locations

The four complete runs are:

- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_baseline_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_baseline_eager_4096_256/`

Complete union comparisons are under this directory. Each subdirectory
contains `kernel_comparison.csv`, `operator_comparison.csv`,
`shape_dtype_comparison.csv`, `cpu_operator_type_comparison.csv`,
`comparison_summary.json`, and `comparison.md`:

- `plugin_vs_native_graph/`
- `plugin_vs_native_eager/`
- `plugin_graph_vs_eager/`
- `native_graph_vs_eager/`
