# megatron-iluvatar

天数（Iluvatar / CoreX）上 MiniCPM 单机 8 卡对照：厂商 Megatron 基线、Python DGC、CoreX kernel DGC。

对应 ts-tg200：

- 脚本与短摘要：`/home/yaowenxuan/ff/pretrain/minicpm`
- 源码：`/home/yaowenxuan/ff/tg200/{ixmegatron,ixmegatron-speedup,ixmegatron-speedup-kernel,dgc-ops}`

不提交 dataset、ckpt、`runs/` 里的完整 profiler trace 和 data_cache。

## 目录

```text
Megatron-LM/                 厂商 ixmegatron（无 DGC）
Megatron-LM-dgc/             ixmegatron-speedup（Python DGC）
Megatron-LM-dgc-corex/       ixmegatron-speedup-kernel（DGC + CoreX 算子）
dgc-ops/                     CoreX DGC 算子源码与微基准（不含 .so / build）
scripts/minicpm/             单机 8 卡启动脚本 + pretrain 入口
profiler/minicpm/            ALL_COMM CSV、短 profiler 摘要、run_config
```

## 对照结果（ALL_COMM，profiler 开）

单机 8 卡，TP=1 / PP=4 / DP=2，16 层，seq=64，global batch=32，10 iter。

| 对照 | 脚本 | baseline_ms | optimized_ms | decrease_pct | verdict |
|---|---|---|---|---|---|
| Python DGC | `train_tg200_8gpu_minicpm.sh` · `...-dgc.sh` | 7926.41 | 6284.50 | 20.71 | PASS |
| CoreX DGC | 同一 baseline · `...-dgc-corex.sh` | 7926.41 | 6155.13 | 22.35 | PASS |

明细：`profiler/minicpm/minicpm-dgc-result.csv`、`minicpm-corex-kernel-result-20260808_053927.csv`。

## 跑脚本

脚本默认指向本仓库内对应 Megatron 目录。数据与 tokenizer 仍是当时集群路径；卡号 `6-13` 是 ts-tg200 那次 8 卡切片，换机器时覆盖 `DATA_PATH` / `TOKENIZER_PATH` / `SELECTED_GPUS`，不要改 TP / PP / DP。

```bash
bash scripts/minicpm/train_tg200_8gpu_minicpm.sh
bash scripts/minicpm/train_tg200_8gpu_minicpm-dgc.sh
bash scripts/minicpm/train_tg200_8gpu_minicpm-dgc-corex.sh
```

CoreX 算子需在 `dgc-ops/` 按该目录 README 编译出扩展后再跑第三套脚本。算子微基准见 `dgc-ops/RESULTS.md`。
