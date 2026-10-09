# LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit

## 方法

使用 MLX-LM 0.31.3 的 LoRA SFT：末 16 层、rank 8、scale 4，沿用上游可转换投影。
当前 Qwen3 包含 attention 的 q/k/v/o 和 MLP 的 gate/up/down 投影。
MLX 的 scale 直接乘 adapter 输出，不能当作其他框架的 LoRA alpha。
BF16 基座使用 LoRA；量化基座使用 QLoRA。先反量化合并，再导出 MLX 4-bit 权重。
当前默认 batch 1、累积 16、长度 1536、lr 1e-6，开启梯度检查点。
使用 Adam；默认固定学习率，`--cosine-schedule --warmup-updates N` 启用上游调度。
N 的单位是 optimizer update，cosine 最低学习率为初始值的十分之一。
实现通过上游 `train_model` 接入 JSONL 指标回调。

`iters` 和看板 step 是微批次次数，累积 4 时每 4 次执行一次 optimizer update。
100 步试跑只包含 25 次参数更新、100 条训练样本，不代表遍历完整训练集。
上游在第 N 个微批次更新前进行验证，因此回调记录为 N-1；保存则发生在更新后。
默认验证随机选取 25 个 batch，不同验证点不是同一批样本。
检查点选择应另用固定验证集或 `--val-batches -1` 全量验证，不能只比较看板局部 loss。

数据使用 Qwen 聊天模板，最后一个用户消息加入 `/no_think`。预处理检查完整序列与
prompt token 长度，给 assistant 训练目标保留空间，去除完全重复的对话后按 seed 划分。
推理关闭 thinking。情感数据中的内部推理不作为最终回复训练目标。
当前 `mask_prompt=True` 只监督每条对话最后一个 assistant 回复；此前各轮也属于 prompt。
精确去重不保证同一角色、近似对话或来源分组在训练与验证之间隔离。

数据与推理共用 `qwen_messages`：可选首个 system，user/assistant 交替；末轮 user
含 `/no_think`，assistant 内容以空 `<think>\n\n</think>\n\n` 开头。预处理剔除不合法轮次，
训练入口在加载权重前验证结构。空 think 前缀对应 Qwen 官方非 CoT 示例。
Qwen tokenizer 本身也会自动给最后一轮回复补此前缀；2026-10-08 仅根据原始字段
判定旧训练目标缺少空 think 的结论有误，已通过实际受监督 token 检查撤回。
旧记录中的连续 assistant 确实会导致 prompt offset 错位；旧推理也没有 `/no_think`。
空前缀显式规范化不能替代轮次验证，reasoning 能力保留仍需独立评测。

续训仅恢复 adapter 权重；不恢复 optimizer、随机数或数据游标。
每次训练与续训写入新的输出目录，看板步数与模型卡统计对应所选目录的记录。
拒答探针采用字符串匹配；A/B 人评匹配人设、生成预算与 seed。
采样、评测和模型卡示例显式使用 Qwen 推荐的非思考采样：temperature 0.7、top-p 0.8、
top-k 20。对照实验记录用户标记、提示词、采样设置、seed 与生成预算。
当前实现 SFT 与偏好收集，DPO、GRPO、权重消融尚未实现。
`Uncensored` 是目标定位，拒答变化和通用能力仍需独立验证。

## 参数对照

`python -m lingxi.tune --out runs/qwen3-tune-01` 从同一基座分别训练 4 个候选：
lr 为 1e-4 / 2e-5，MLX scale 为 20 / 4，固定 rank 8、最后 16 层、200 微批次、
50 次参数更新、batch 1、累积 4 与 seed 42。每个候选保存独立日志和权重。
使用 seed 42 预选相同 128 条验证记录，在保存后的实际权重上计算 token 加权 loss；
记录验证索引与数据 SHA256。回复对照固定 10 道题、384-token 上限和每题 seed。
重复 12-gram 与图片链接只作诊断，选择还需审阅事实、角色和指令遵循。
该验证子集用于选参，不是独立外部能力基准；候选胜出不代表全局最优。

2026-10-09 使用重新规范化的数据：训练 22,591 条、验证 461 条，过滤后总计 23,052 条。
逐条核对模板和最后回复的监督 token。原始数据和旧划分留存，新数据另存目录。
第一轮四组 loss 分别为 1.921、1.820、1.836、1.860，基座为 3.341；均出现回复
过短、循环或基础能力退化，未据此启动完整训练。第二轮将 lr 降至 5e-6，比较
累积 4/16 与 rank 8 / scale 4、rank 16 / scale 2，继续检查原始回复。
相同 256 微批次下，三组固定 loss 为 1.946、2.591、2.611。累积 4 的候选仍有
事实退化；两个累积 16 候选保留了乘法、JSON 与昼长解释，没有明显长段循环。
rank 16 未改善 loss，后续使用 rank 8 / scale 4 检查更长预算和调度：

```bash
python -m lingxi.tune --data data/processed-qwen3 \
  --out runs/qwen3-tune-03/lr5e-06-rank8-scale4-acc16-cosine \
  --iters 1024 --learning-rate 5e-6 --lora-rank 8 --lora-scale 4 \
  --grad-accumulation-steps 16 --cosine-schedule
```

该轮为 64 次更新、warmup 3 次，再衰减至初始 lr 的十分之一。短程与长程训练
预算不同，不能将它们解释为严格的调度消融；角色表达问题仍需单独评测。
该轮实际固定 loss 为 2.003，角色题 4、6 出现循环，因此未用于整轮训练。
继续以相同 1,024 微批次、rank 8、scale 4、累积 16 和调度对照 lr 1e-6。
训练目标中的重复阈值命中 407 条，中文占比较高的两条主要是引用标记；
步骤、代码和引用可能正常重复，未据此批量改动数据。
lr 1e-6 的同预算结果为 3.098，未出现此前的明显角色循环，乘法、JSON 和昼长解释
保留；仍有模板化措辞、轻微重复与算术题输出格式不严格的问题。选择该组作为本次
整轮训练参数，结论限定于已测试的候选与 seed；不是全局最优或独立能力结论。
整轮为 22,592 微批次、1,412 次更新，warmup 70 次更新，cosine 最低 lr 1e-7。
一轮覆盖 22,591 条训练记录，再多一个微批次以完成最后一次累积更新。
完整流程按一轮数据计算微批次，并对齐梯度累积；训练后全量验证、检查回复，
通过诊断再合并与导出，重新加载验证实际 4-bit 层并评测量化模型。导出不上传 HF。

## 上游依据

- [MLX-LM v0.31.3](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3)
- [LoRA 训练与内存设置](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/LORA.md)
- [训练入口](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/lora.py)
- [LoRA 投影选择](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/tuner/utils.py)
- [梯度累积、验证与 prompt mask](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/tuner/trainer.py)
- [Qwen3 推理与采样建议](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3 非思考模式微调](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md)
- [Qwen3 tokenizer 模板](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json)
