# LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit

## 方法

使用 MLX-LM 0.31.3 的 LoRA SFT：末 16 层、rank 8、scale 20，沿用上游可转换投影。
BF16 基座使用 LoRA；量化基座使用 QLoRA。最终合并并导出 MLX 4-bit 权重。
默认 batch 1、累积 4、长度 1536、lr 1e-4，开启梯度检查点。这些是起始配置。
实现通过上游 `train_model` 接入 JSONL 指标回调。

数据使用 Qwen 聊天模板，最后一个用户消息加入 `/no_think`。预处理检查完整序列与
prompt token 长度，给 assistant 训练目标保留空间，去除完全重复的对话后按 seed 划分。
推理关闭 thinking。情感数据中的内部推理不作为最终回复训练目标。

续训仅恢复 adapter 权重；不恢复 optimizer、随机数或数据游标。
看板拼接多段指标的累计步数是估计，原始迭代数仍保留。
拒答探针采用字符串匹配；A/B 人评匹配人设、生成预算与 seed。
当前实现 SFT 与偏好收集，DPO、GRPO、权重消融尚未实现。
`Uncensored` 是目标定位，拒答变化和通用能力仍需独立验证。

## 上游依据

- [MLX-LM v0.31.3](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3)
- [LoRA 训练与内存设置](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/LORA.md)
- [训练入口](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/lora.py)
- [LoRA 投影选择](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/tuner/utils.py)
- [Qwen3 非思考模式微调](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md)
- [Qwen3 tokenizer 模板](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/tokenizer_config.json)
