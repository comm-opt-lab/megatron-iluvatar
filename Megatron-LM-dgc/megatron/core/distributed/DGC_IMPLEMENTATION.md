# DGC (Deep Gradient Compression) 实现说明

本文档记录了在 `megatron/core/distributed/` 中新增的 DGC 梯度稀疏化压缩功能的设计、
代码改动点、使用方法和已知限制，方便后续查阅与维护。这是本地新增功能，**不是**
NVIDIA 上游 Megatron-Core 的一部分。

参考论文：Lin et al., *"Deep Gradient Compression: Reducing the Communication
Bandwidth for Distributed Training"*, ICLR 2018. https://arxiv.org/abs/1712.01887

## 1. 需求

- 是否开启梯度稀疏度压缩：可通过开关控制（`--dgc-enabled`）。
- 压缩后保留多少梯度参数：可配置（`--dgc-density`，保留比例）。
- 要求是**真正减少通信字节数**的稀疏通信，而不是“仍走 dense all-reduce、只是把值
  置零”的轻量级实现（这是与用户确认过的设计选择，见下文“架构权衡”）。

## 2. 架构权衡（已与用户确认）

原始代码中数据并行梯度同步全部基于**连续显存 buffer + NCCL all-reduce /
reduce-scatter**（`megatron/core/distributed/param_and_grad_buffer.py` 的
`_ParamAndGradBucketGroup.start_grad_sync`），并与 FP8 参数收集、分布式优化器分片
（reduce-scatter）、SDMA 通信、多 DistOpt 实例等特性深度耦合。

真正做到“减少通信字节数”的 DGC 需要把 dense all-reduce 替换成“稀疏索引 + 数值”的
通信原语，这与上述所有特性都有冲突。完整支持所有路径的稀疏通信工作量巨大且风险高，
因此本实现的范围限定为：

- **支持**：纯 DDP all-reduce 路径（`use_distributed_optimizer=False`、
  `num_distributed_optimizer_instances=1`），同步通信（`overlap_grad_reduce=False`），
  非 collective-内平均（`average_in_collective=False`）。
- **不支持（会在配置校验阶段直接报错，而不是静默退化）**：分布式优化器
  reduce-scatter 路径、SDMA、多 DistOpt 实例、`overlap_grad_reduce=True`、
  `average_in_collective=True`。这些路径下稀疏 reduce-scatter / 异步流水线的正确性
  远比 all-reduce 路径复杂，留作后续工作。

## 3. 算法实现

完整实现 DGC 论文中的 top-k 稀疏化 + 动量修正（momentum correction）+ 误差反馈
（error feedback），每个 bucket 独立维护两个持久化的状态 buffer：

- `velocity`（动量/速度 buffer）：`v_t = momentum * v_{t-1} + g_t`
- `residual`（本地累积/误差反馈 buffer）：`u_t = u_{t-1} + v_t`

每一步：

1. 用上述递推更新 `velocity` 与 `residual`。
2. 在 `residual` 上按绝对值取 Top-K（`k = max(1, int(numel * density))`，K 在给定
   bucket 大小下是常数，因此每个 DP rank 选出的元素个数一致，便于用固定形状的
   `all_gather` 通信）。
3. 把被选中位置的 `velocity` / `residual` 清零（已发送的梯度信息重新从 0 开始累积，
   未选中位置继续累积，等待未来某一步被选中——这就是“误差反馈”补偿稀疏化丢失的
   梯度信息的机制）。
4. 用 `torch.distributed.all_gather` 在数据并行组内交换 `(indices, values)`
   （而不是 dense 的 `all_reduce`/`reduce_scatter`）。
5. 每个 rank 在本地用 `scatter_add_` 把所有 rank 发来的稀疏更新加总，重建出
   “全局求和后”的稀疏梯度，写回 `bucket.grad_data`，供后续优化器步骤使用。

代码位置：

- `megatron/core/distributed/dgc.py`
  - `DGCConfig`：压缩参数（density / momentum / min_numel_to_compress）。
  - `DGCCompressor.compress(grad)`：执行上述步骤 1-3，返回 `(indices, values)`。
    状态以 `id(bucket.grad_data)` 为 key（而不是 `bucket.bucket_id`，因为
    `bucket_id` 只在单个 `_ParamAndGradBuffer` 内唯一，跨 buffer 时可能重复，例如
    FP8 场景下 fp8/bf16 两个 buffer 被合并进同一个 bucket group）。
- `megatron/core/distributed/param_and_grad_buffer.py`
  - `_ParamAndGradBucketGroup.__init__`：当 `ddp_config.dgc_enabled` 为真时，构造
    `self.dgc_compressor`。
  - `_ParamAndGradBucketGroup.start_grad_sync`：在 `check_grads` 之后插入分支，若
    `self.dgc_compressor is not None` 则调用 `_start_grad_sync_dgc()` 并直接
    `return`（不走原有 dense all-reduce / reduce-scatter 逻辑）。
  - `_ParamAndGradBucketGroup._start_grad_sync_dgc`（新方法）：实现上述步骤 4-5，
    对小于 `dgc_min_numel_to_compress` 的 bucket 直接退化为 dense `all_reduce`
    （索引+数值的开销对小 bucket 不划算）。
- `megatron/core/distributed/distributed_data_parallel_config.py`
  - 新增字段 `dgc_enabled` / `dgc_density` / `dgc_momentum` /
    `dgc_min_numel_to_compress`。
  - `__post_init__` 中新增校验：开启 DGC 时必须满足第 2 节列出的兼容性要求，
    否则在构造 `DistributedDataParallelConfig` 时就会 `assert` 失败（fail-fast，
    而不是训练中途出现隐蔽的正确性问题）。
- `megatron/training/arguments.py`
  - 新增命令行参数 `--dgc-enabled` / `--dgc-density` / `--dgc-momentum` /
    `--dgc-min-numel-to-compress`，字段名与 `DistributedDataParallelConfig` 完全
    一致，因此 `training.py` 中已有的
    `for f in dataclasses.fields(DistributedDataParallelConfig): kwargs[f.name] = getattr(args, f.name)`
    会自动把这些参数透传进 DDP 配置，无需改动 `training.py`。
- `run-scripts/pretrain_9g70b_bash.sh`
  - 新增 `DGC_ARGS_ENABLED`（示例参数块，默认未生效）与 `DGC_ARGS=""`
    （默认关闭，已加入 `CMD`）。如需开启，把 `DGC_ARGS=""` 改为
    `DGC_ARGS="${DGC_ARGS_ENABLED}"`。

## 4. 使用方法

```bash
--dgc-enabled                       # 开启 DGC 稀疏梯度通信（默认关闭）
--dgc-density 0.01                  # 每个 bucket 保留/通信 1% 的梯度元素（99% 稀疏度）
--dgc-momentum 0.9                  # 动量修正系数，默认 0.9
--dgc-min-numel-to-compress 16384   # 小于该元素数的 bucket 退化为 dense all-reduce
```

开启前提（否则 `DistributedDataParallelConfig.__post_init__` 会直接报错）：

- 不能同时使用 `--use-distributed-optimizer`
- 不能同时使用 `--overlap-grad-reduce`
- 不能同时使用 `--ddp-average-in-collective`
- DistOpt 实例数必须为 1（即不设置多实例分布式优化器）

## 5. 通信量与收益的权衡

设 bucket 元素数为 `N`，DP world size 为 `P`，density 为 `d`，则：

- Dense ring all-reduce 通信量：约 `O(2 * (P-1)/P * N)` 个元素。
- DGC（基于 all_gather 的稀疏方案）通信量：约 `O(2 * k * P)` 个元素，其中
  `k = max(1, N * d)`（indices + values 各占一份，且 `all_gather` 是把本地的 k 个
  元素发送给所有其它 P-1 个 rank，因此与 P 成正比，而不是像 reduce-scatter 那样
  分摊）。

因此 DGC 在 `d * P << 1`（即 density 远小于 1/P）时才能真正降低通信量；当 DP
world size 较大、density 又设置得不够小时，`all_gather` 方案的通信量甚至可能超过
dense ring all-reduce。这是基于 `all_gather` 的稀疏方案的固有特性（与原始 DGC 论文
使用 MPI `Allgatherv` 的方式一致），实际使用时需要结合 `dp_size` 选择合适的
`--dgc-density`。

索引张量目前使用 `int64`（`torch.scatter_add_` 的硬性要求），相比理论上可行的
`int32` 索引会多一倍的 index 通信开销；如果后续需要进一步压缩通信量，可以在通信前
转换为 `int32`、接收后转回 `int64` 再 `scatter_add_`（局部转换，不影响算法正确性），
这是一个可选的后续优化点，未在本次改动中实现。

## 6. 已知限制 / 后续工作

1. 不支持分布式优化器（reduce-scatter）路径——稀疏 reduce-scatter 需要让每个 rank
   只重建出自己负责的那一个分片，而不是全量梯度，索引集合与分片边界不天然对齐，
   实现复杂度高，本次未实现。
2. 不支持 `overlap_grad_reduce=True`（异步通信与计算重叠）——当前 `_start_grad_sync_dgc`
   是同步实现（每个 bucket 依次发起 `all_gather` 并立即等待完成）。
3. 不支持多 DistOpt 实例（`num_distributed_optimizer_instances > 1`）和 SDMA 通信路径。
4. 未实现 DGC 论文中的 warm-up 阶段（训练初期逐步从低稀疏度过渡到目标稀疏度）和
   压缩前的局部梯度裁剪（gradient clipping before compression）。当前只暴露了
   `density`（保留比例）和 `momentum` 两个核心可调参数，满足“是否开启 + 保留比例
   可配置”的需求；如需更贴近论文的完整方案可以后续在 `DGCCompressor` 中加入。
5. `momentum correction` 的理论推导基于普通 SGD（无 momentum 的情况下退化为单纯的
   误差反馈/残差累积）；当外层优化器是 Adam 等自适应优化器时，DGC 的 momentum
   correction 不再有论文中那样严格的理论保证，但残差累积（error feedback）部分依然
   是社区中通用的稀疏化补偿做法。

## 7. 性能统计：对比 DGC 开启前后的效果

### 7.1 需要添加的参数

要统计 DGC 开启前后的计算 / 通信开销和单迭代耗时，除了 DGC 本身的开关参数外，
还需要加上 Megatron 自带的计时与吞吐量统计参数（只影响日志输出，不改变训练行为，
DGC 关闭时也不引入任何额外开销）：

```bash
--timing-log-level 1        # 开启一次性算子级别计时（包含 all-grads-sync 以及
                             # DGC 新增的 dgc-compress/dgc-comm/dgc-reconstruct，
                             # 默认 0 时这些计时器是空操作的 DummyTimer，不会被打印）
--timing-log-option minmax  # 按 rank 输出 (min, max) 而不是单一数值，避免某个 rank
                             # 不具代表性（默认值就是 minmax，可不显式传）
--log-throughput             # 额外输出每 GPU 的 TFLOP/s 吞吐量，用于比较计算效率/MFU
--log-interval 1             # 每个 iteration 都打印一次（脚本中已默认设置），
                             # 打印窗口越多，下面 compare_dgc.sh 的均值统计越平滑
```

`run-scripts/pretrain_9g70b_bash.sh` 中已经把这些参数封装成 `PROFILE_ARGS`
（由环境变量 `PROFILE_ENABLED` 控制，默认开启），无需手动添加。

### 7.2 一次迭代里"计算"与"通信"分别对应哪些 timer

Megatron 每 `--log-interval` 个迭代打印一次 `(min, max) time across ranks (ms):`
表格，加上单条迭代汇总日志行 `elapsed time per iteration (ms): ...`。各部分耗时与
"计算 / 通信"的对应关系：

| 类别 | Timer 名称 | 说明 |
|---|---|---|
| 计算 | `forward-compute` | 前向计算 |
| 计算 | `backward-compute` | 反向计算 |
| 通信（梯度同步总耗时） | `all-grads-sync` | DGC 关闭时 = dense all-reduce 耗时；DGC 开启时 = 下面三段之和，因此**可以直接和基线对比** |
| 计算（DGC 新增） | `dgc-compress` | 本地动量修正 + 误差反馈 + top-k 选择 |
| 通信（DGC 新增） | `dgc-comm` | 稀疏 `(indices, values)` 的 `all_gather` |
| 计算（DGC 新增） | `dgc-reconstruct` | 本地 `scatter_add_` 重建全量梯度 |
| 通信 | `params-all-gather` | 仅分布式优化器路径才有，DGC 不支持该路径，因此恒为空 |
| 整体 | `elapsed time per iteration (ms)` | 单次迭代的端到端总耗时（包含上面所有阶段 + 优化器步骤等） |
| 整体 | `throughput per GPU (TFLOP/s/GPU)` | 每 GPU 算力吞吐量，可用来判断通信瓶颈是否被压缩掉 |

`dgc-compress` / `dgc-comm` / `dgc-reconstruct` 三个 timer 由
`_ParamAndGradBucketGroup._dgc_timed`（`param_and_grad_buffer.py`）实现：在每个
阶段前后插入 `torch.cuda.synchronize()`，确保统计的是真实 GPU 执行时间而不是
CPU 异步下发的瞬时耗时——这是专门为性能分析准备的，会引入额外的同步开销，因此只在
`dgc_enabled=True` 时生效，不影响 DGC 关闭时的基线性能。这些 timer 通过
`ddp_config.dgc_model_config`（在 `distributed_data_parallel.py` 的
`DistributedDataParallel.__init__` 中惰性绑定到训练循环里的 `config.timers`）取得
`Timers` 对象；`megatron/training/training.py` 中的 `timers_to_log` 列表也在
`args.dgc_enabled` 为真时自动加入这三个新名字，使其出现在每个 `--log-interval`
打印的表格里（以及 `--log-timers-to-tensorboard` 写入 TensorBoard 的内容里）。

### 7.3 自动化对比脚本：`run-scripts/compare_dgc.sh`

```bash
# 单机示例（按实际集群环境变量替换 node_num/gpu_num/inputtest/inputmodel）
node_num=1 gpu_num=8 inputtest=/data/oscar inputmodel=/data/llama2 \
    bash run-scripts/compare_dgc.sh 0.01     # 0.01 = DGC density，可省略，默认 0.01
```

该脚本依次以 `DGC_ENABLED=0`（基线）和 `DGC_ENABLED=1 DGC_DENSITY=<density>`
两种配置调用 `pretrain_9g70b_bash.sh`，把完整输出分别重定向到按节点区分的
`run-scripts/compare_logs/dense.node<RANK>.log` 和
`run-scripts/compare_logs/dgc_density<density>.node<RANK>.log`，只在 `RANK=0`
（master 节点）上用 `awk` 汇总解析所有节点的日志（glob 匹配 `*.node*.log`），
按"每个打印窗口取 (min,max) 中的 max，再对所有节点、所有打印窗口取平均"的方式
得到每个指标的代表值，最终打印一张并排对比表，并计算"单迭代耗时变化百分比"和
"梯度同步耗时变化百分比"。

注意事项：
- **多机场景下日志必须按节点区分文件名**（已修复，曾经是一个真实踩到的坑）：
  本仓库代码目录在 Lustre 等共享文件系统上对所有节点可见且路径相同，如果日志
  文件名不带节点标识，多机同时跑（K8s 多 worker pod 场景下，每个 pod 都在执行
  同一份 `compare_dgc.sh`）会导致多个节点的输出竞争写同一个文件、互相覆盖，
  最终只能看到部分节点的数据，且无法分辨缺的是因为没跑还是被覆盖了。现在文件名
  按 `$RANK`（与 `pretrain_9g70b_bash.sh` 里 `NODE_RANK=${RANK:=0}` 同一个变量）
  区分，且只有 `RANK=0` 节点负责解析打印汇总表（其余节点跑完训练直接退出），
  `RANK=0` 节点在解析前会 `sleep 10` 等其它节点把日志 flush 完，这是尽力而为的
  软同步，不是强同步——如果各节点训练耗时差异很大或日志没及时落盘，可以加大这个
  等待时间，或者等所有节点确认跑完后手动重新执行解析部分（把脚本里 `sleep 10`
  之后的代码单独摘出来跑一遍）。
- 多机场景下仍然需要按平时启动 `pretrain_9g70b_bash.sh` 的方式（外部作业调度
  系统——这里是 Kubernetes，每个 worker pod 一个节点——在每个节点分别设置
  `RANK`/`MASTER_ADDR` 等环境变量），由调度系统在所有节点上同时调用
  `compare_dgc.sh`；脚本本身不做跨节点的 pod 编排/调度。
- 训练迭代数较少（脚本里 `TRAIN_ITERATIONS=40`）时，前几次迭代通常因 CUDA/NCCL
  warm-up 偏慢，会拉高均值；如需更精确的稳态对比，可以加大 `TRAIN_ITERATIONS`，
  或修改 `compare_dgc.sh` 中的 awk 脚本跳过前 N 个打印窗口。
- 对比结果强依赖 `--dgc-density` 与数据并行规模 `dp_size`（参见第 5 节的
  `O(2*k*P)` vs `O(2*(P-1)/P*N)` 权衡分析）：`dp_size` 越大，需要越小的 density
  才能让 DGC 在通信耗时上体现优势；如果对比结果显示 DGC 开启后 `all-grads-sync`
  反而变慢，通常意味着当前 `density * dp_size` 不够小。注意如果 `TP * PP == 总
  GPU 数`（即 `dp_size == 1`），数据并行组退化为单 rank，dense 和 DGC 的通信都是
  no-op，看不出任何通信耗时差异，这种配置只能用来验证显存开销和正确性，无法用来
  评估 DGC 的通信收益。

## 8. 显存开销与 OOM 排查

### 8.1 DGC 会额外占用多少显存

`DGCCompressor` 为每个被压缩的 bucket 维护 **与该 bucket 梯度 buffer 等大** 的
误差反馈状态，且在整个训练过程中常驻显存（不会被释放）：

- `residual`（本地累积 buffer）：恒定存在，1 倍 bucket 显存。
- `velocity`（动量修正 buffer）：仅当 `--dgc-momentum > 0`（默认值 0.9）时存在，
  再加 1 倍 bucket 显存。`--dgc-momentum 0` 时会跳过分配，只保留 `residual`。

也就是说，DGC 开启后每个 rank 的显存会比 dense 基线**额外增加**：

```
额外显存 ≈ Σ(被压缩 bucket 的元素数 × dtype 字节数) × (2，若 momentum > 0；否则 1)
```

由于 `overlap_grad_reduce=False`（DGC 的强制要求，见第 2 节）会让每个
`_ParamAndGradBuffer` 退化成一个超大 bucket（bucket_size 被设为
`None`/无穷大，参见 `distributed_data_parallel.py` 中
"Set bucket_size to infinity if overlap_grad_reduce is False"），实际相当于
**该 PP/TP 切片本地全部梯度 buffer 的大小**被翻倍（momentum=0）或triple
（momentum>0，原 grad_data + 额外 residual + 额外 velocity = 3 倍）。对于已经
接近显存上限的大模型（plain DDP，未使用 `--use-distributed-optimizer`，因为
DGC 当前不支持分布式优化器路径，见第 6 节限制 1），这笔额外开销很容易直接导致
`CUDA out of memory`。

**这就是 DGC 在显存紧张环境下最常见的失败模式**：dense 基线本来就跑在显存边缘
（例如已用 58.68 GiB / 63.59 GiB），DGC 在此基础上需要至少再多 1 倍（momentum=0）
甚至 2 倍（momentum>0）梯度 buffer 大小的显存，必然 OOM。

为了让这个问题尽早暴露而不是训练到一半才崩溃，现在的实现做了两处改动：

1. `DGCCompressor.preallocate()` 在 `_ParamAndGradBucketGroup.__init__`（即 DDP
   构造阶段，早于优化器状态分配、早于任何前向/反向）就**提前分配**好
   residual/velocity buffer，而不是像之前那样等到第一次反向传播时才惰性分配——
   如果显存真的不够，会在训练刚启动时就以清晰的报错失败，而不是在某次反向传播
   中以一个看起来无关的 stack trace 失败。
2. 同一处会打印一条 INFO 日志（通过 `log_on_each_pipeline_stage`），明确给出
   "DGC enabled: error-feedback state ... will use an extra X.XX GiB of GPU
   memory per rank"，建议在显存不够时尝试的两个参数（见下）。

### 8.2 OOM 时可以尝试的参数（按推荐顺序）

1. **`--dgc-momentum 0`**：把 `velocity` buffer 去掉，额外显存从 2 倍降到 1 倍
   bucket 大小。代价是退化为纯粹的"残差累积/误差反馈"（不做动量修正），DGC 论文
   中针对 SGD 的动量修正本来就只在使用 SGD 时有完整理论保证（见第 6 节限制 5），
   对 Adam 等自适应优化器影响有限，通常是显存紧张时的首选。
2. **调大 `--dgc-min-numel-to-compress`**：因为 `overlap_grad_reduce=False` 时
   一个 buffer 通常只有一个大 bucket，调大这个阈值会让该 bucket 整体跳过压缩、
   退化为 dense all-reduce（不再分配 residual/velocity），相当于对某些 PP stage
   单独关闭 DGC——可以用于只在显存有富余的 stage 上开启 DGC。
3. **错误信息里 PyTorch 自带的建议**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   可以缓解显存碎片化问题，与 DGC 本身无关，可以叠加使用（仅当同时使用
   `nccl_ub=True` 时两者不兼容，详见 `distributed_data_parallel_config.py` 中
   的校验，本脚本未开启 `nccl_ub`，不受影响）。
4. **从其他地方腾显存**：增大 `--pipeline-model-parallel-size`（让每个 stage
   的本地参数更少，从而梯度 buffer 和 DGC 状态都更小）、开启激活值重计算
   （activation recomputation）、减小 `--micro-batch-size`，或者——如果业务上能
   接受——评估是否真的需要在"显存已经打满的 plain DDP 大模型 + DGC"这个组合下
   运行；DGC 当前实现与 `--use-distributed-optimizer`（ZeRO 风格的优化器状态分片，
   通常能省下大量显存）互斥，是本实现待补的能力（见第 6 节限制 1），如果显存
   是核心瓶颈，分布式优化器对显存的帮助通常比 DGC 压缩通信带来的额外开销更值得
   优先考虑。

### 8.3 实测验证

在一次 4 节点 × 5 GPU（`TP=2, PP=10, dp_size=1`）的实测中，8.1 节的日志预估与
实际显存增量完全吻合：日志打印"额外 3.63 GiB"的 stage，dense→DGC 实测内存从
19587 MB 涨到 23300 MB（差值 3713 MB ≈ 3.63 GiB）；日志打印"额外 4.75 GiB"的
stage（embedding/output 层所在 stage），实测从 26093 MB 涨到 30954 MB（差值
4861 MB ≈ 4.75 GiB）。说明 8.1 节的公式和日志预估是准确的，可以直接当作显存
规划依据。

同一次实测中还发现了一个容易误诊为"DGC 内存泄漏"的假象：当时只有 4 个节点中的
2 个节点在日志里出现了数据，看起来像是另外 2 个节点显存异常飙升到 59 GiB+ 而
OOM；但排查后发现根本原因是 7.3 节描述的"多机日志文件名不带节点标识、共享文件
系统上互相覆盖"的 bug，并不是 DGC 真的在那 2 个节点上多用了几十 GB 显存——日志
缺失节点的真实情况实际是未知的（数据被覆盖，不是没发生 OOM）。这个 bug 已经
在 `compare_dgc.sh` 里修复（日志按节点拆分），如果显存问题在使用最新版脚本后
仍然复现，才需要继续按 8.2 节排查；如果只是因为日志覆盖看起来很吓人，重新跑一遍
通常就能看到完整、一致的数据。
