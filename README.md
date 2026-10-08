# LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit

Chinese roleplay and emotional-support post-training for Qwen3-1.7B, with MLX LoRA SFT and a 4-bit export on Apple Silicon.

灵犀：面向中文角色扮演与情感对话的 Qwen3-1.7B 后训练实验。
`Uncensored` 是目标定位，拒答变化和通用能力需要分别评测。
当前实现 SFT、拒答探针与 A/B 偏好收集；DPO、GRPO 尚未实现。
训练阶段可使用 BF16 基座，名称中的 `4Bit` 指最终 MLX 量化产物。
方法与上游依据见 [METHOD.md](METHOD.md)。

## 结构

```text
prepare.py        数据转换、token 过滤、去重与划分
train.py          MLX LoRA SFT、配置与指标记录
persona.py        模型名称与默认人设
sample.py         终端对话
eval.py           拒答探针与 A/B 人评
dashboard.py      本地看板服务
dashboard.html    看板页面，无前端构建工具
publish.py        合并、量化、模型卡与可选 HF 上传
probes.jsonl      拒答探针
questions.md      A/B 题集
```

数据、基座、运行记录和导出权重分别存入 `data/`、`models/`、`runs/`、`dist/`，留在 Git 外。
七个 Python 入口直接运行，无包结构。问题与改动使用本仓库的 GitHub Issues。

## 准备

需要 Apple Silicon Mac 和 Python 3.13；依赖固定在 `requirements.txt`。
MLX 后端的迁移范围是其他 Apple Silicon 机器。

```bash
git clone https://github.com/YuanyuanMa03/lingxi-qwen3-1.7b-uncensored.git
cd lingxi-qwen3-1.7b-uncensored
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir models/Qwen3-1.7B
hf download usamakenway/unified-uncensored-qwen-chatml-sft --repo-type dataset \
  --revision 4344a1318b78efc511c1d60b7d54ef2e137e81c3 \
  --include 'data/train-*.parquet' --local-dir data/uncensored-chatml
hf download shibing624/roleplay-zh-sharegpt-gpt4-data --repo-type dataset \
  --revision 9c8bd488eb5cb4489a8c2c31518bd0d64154578d \
  --include 'sharegpt_formatted_data-*.jsonl' --local-dir data/roleplay-zh
hf download AngelWarmSmile123/deep-emotional-support-zh --repo-type dataset \
  --revision e72ab43f7ac4a60e723b7550f9c30de5a9f20d78 \
  --include 'emotional_support.jsonl' --local-dir data/emotional-support-zh
```

ChatML 来源规模较大，下载前检查可用磁盘。已有资源可放入相同相对目录。
所有脚本以仓库根目录解析相对路径；模型也可通过 `--model` 使用 HF ID。
换机器时携带实际模型、数据和运行目录，检查符号链接的目标是否可用。

## 训练与看板

```bash
python prepare.py
python train.py
python dashboard.py
```

打开 `http://127.0.0.1:8765`。默认运行目录为 `runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/`。
预处理保留 system prompt，添加 `/no_think`，使用基座聊天模板过滤过长或目标 token 不足的样本，
精确去重后固定 seed 划分。已有划分拒绝覆盖；新的预处理使用 `--out data/processed-v2`。
精确去重不等于按人物或近似文本隔离，验证集只是同分布验证。

默认末 16 层、rank 8 / scale 20、batch 1、累积 4、长度 1536、lr 1e-4，开启梯度检查点。
这些是起始配置，先用 `--iters` 做短程检查，再依据本机内存调整。
每次训练保存配置快照、数据统计与 `metrics.jsonl`。已有指标的目录必须显式续训：

```bash
python train.py \
  --resume runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters/0000600_adapters.safetensors \
  --iters 100
```

请按实际存在的检查点选择文件。MLX 续训加载 adapter 权重，`--iters` 是本次新增步数；
不恢复 optimizer、随机数或数据游标。看板多段日志的累计步数是拼接估计。

## 评测与导出

```bash
python sample.py --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters
python eval.py --label base
python eval.py --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters --label sft
python eval.py --ab --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters
python publish.py --dry-run
python publish.py
```

拒答结果写入运行目录，A/B 偏好对写入 `data/user_dpo.jsonl`。
两模型匹配人设、预算与 seed，并关闭 thinking；探针字符串匹配可能误判，A/B 题集不是独立能力基准。

`--dry-run` 只展示命令与模型卡。导出先生成 `dist/bf16/`，再生成
`dist/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/`。
明确加 `--upload` 才上传 HF，默认目标为 `myy555/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit`；
其他贡献者通过 `--repo your-account/model-name` 指定自己的仓库，凭据由 HF 管理。

## 数据与许可

代码使用 [Apache-2.0](LICENSE)，基座和数据分别遵循来源许可。

| 来源 | 数据卡许可 / 使用方式 |
| --- | --- |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0，基座 |
| [unified-uncensored-qwen-chatml-sft](https://huggingface.co/datasets/usamakenway/unified-uncensored-qwen-chatml-sft) | 混合许可；按行筛选 `source_license=apache-2.0`，选取 Dolphin / OpenHermes / Airoboros |
| [roleplay-zh-sharegpt-gpt4-data](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 数据卡 Apache-2.0，另有上游来源说明 |
| [deep-emotional-support-zh](https://huggingface.co/datasets/AngelWarmSmile123/deep-emotional-support-zh) | 数据卡 CC-BY-NC-SA-4.0 |

仓库公开代码与处理方法，不再分发原始对话。代码许可不代表混合数据或训练权重的许可；
使用与再分发时应核对来源条款，当前默认配方不能宣称为完全商业可用。

## 验证范围

已检查 CLI、数据过滤与统计、A/B 调度、看板 HTTP API、停止状态和无副作用导出预览。
Qwen3-1.7B 在 128 token 的构造短样本上完成过 4 步训练及 4 步权重续载，
并验证移动仓库后继续运行。该检查只证明管线与相对路径可用，不能证明模型质量或完整训练资源需求。
完整合并、量化及导出后质量评测仍待验证，HF 权重尚未通过本仓库发布。
