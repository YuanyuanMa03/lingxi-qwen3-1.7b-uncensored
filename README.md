# LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit

Chinese roleplay and emotional-support post-training for Qwen3-1.7B, with MLX LoRA SFT and a 4-bit export on Apple Silicon.

灵犀：面向中文角色扮演与情感对话的 Qwen3-1.7B 后训练实验。
`Uncensored` 是目标定位，拒答变化和通用能力需要分别评测。
当前实现 SFT、拒答探针与 A/B 偏好收集；DPO、GRPO 尚未实现。
训练阶段可使用 BF16 基座，名称中的 `4Bit` 指最终 MLX 量化产物。
方法与上游依据见 [METHOD.md](docs/METHOD.md)。

## 结构

```text
lingxi/           训练、推理、评测、导出代码与看板页面
tests/            CLI、真实微型模型与浏览器回归测试
docs/             方法说明、拒答探针与 A/B 题集
requirements.txt  运行依赖
requirements-dev.txt  测试依赖
```

数据、基座、运行记录和导出权重分别存入 `data/`、`models/`、`runs/`、`dist/`，留在 Git 外。
代码通过 `python -m lingxi.<模块>` 运行。问题与改动使用本仓库的 GitHub Issues。

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
python -m lingxi.prepare --out data/processed-qwen3
python -m lingxi.tune --full --data data/processed-qwen3 \
  --out runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit \
  --learning-rate 1e-6 --lora-rank 8 --lora-scale 4 \
  --grad-accumulation-steps 16 --cosine-schedule
python -m lingxi.dashboard
```

打开 `http://127.0.0.1:8765`。默认运行目录为 `runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/`。
预处理保留 system prompt，剔除不合法轮次，显式添加 `/no_think` 与空 think 回复前缀，
使用基座聊天模板过滤过长或目标 token 不足的样本，
精确去重后固定 seed 划分。已有划分拒绝覆盖；重新预处理时指定新的 `--out` 目录。
精确去重不等于按人物或近似文本隔离，验证集只是同分布验证。

当前选择末 16 层、rank 8 / scale 4、batch 1、累积 16、长度 1536、lr 1e-6，开启梯度检查点。
上面的完整流程训练一轮，使用 5% warmup 与 cosine，之后全量验证、检查回复并本地导出。
明显循环或基础诊断失败时暂停导出；量化后也重新检查，不上传 HF。
这是本次对照中的稳定候选，角色表达与独立能力仍需评测。新机器先通过 `lingxi.train --iters N`
做短程资源检查，N 对齐梯度累积倍数。
用固定 128 条验证记录和同样的 10 道回复题对照学习率与 scale：

```bash
python -m lingxi.tune --data data/processed-qwen3 --out runs/qwen3-tune-01
```

每组从基座重新开始，输出各自的配置、指标和原始回复。先检查循环、事实与指令遵循，
再比较验证 loss；候选中胜出的配置不代表全局最佳。完整训练预算对齐梯度累积倍数，
`--lora-rank`、`--lora-scale` 可指定低秩参数，`--cosine-schedule --warmup-updates N`
启用调度，N 按参数更新计数。
每次训练保存配置快照、数据统计与 `metrics.jsonl`。输出目录必须不存在或为空。
续训从旧检查点加载权重，写入新的 `--out`，保留旧目录中的全部产物：

```bash
python -m lingxi.train \
  --resume runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters/0000600_adapters.safetensors \
  --out runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit-resume-01 --iters 100
```

请按实际存在的检查点选择文件。MLX 续训加载 adapter 权重，`--iters` 是本次新增步数；
不恢复 optimizer、随机数或数据游标。查看、评测和导出续训产物时，指定新的运行目录或 adapter 路径：

```bash
python -m lingxi.dashboard --run runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit-resume-01
python -m lingxi.publish --run runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit-resume-01 --dry-run
```

## 评测与导出

```bash
python -m lingxi.sample --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters
python -m lingxi.eval --label base
python -m lingxi.eval --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters --label sft
python -m lingxi.eval --ab --adapter runs/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/adapters
python -m lingxi.publish --dry-run
python -m lingxi.publish
```

拒答结果写入运行目录，A/B 偏好对写入 `data/user_dpo.jsonl`。
两模型匹配人设、预算与 seed，并关闭 thinking；探针字符串匹配可能误判，A/B 题集不是独立能力基准。

`--dry-run` 只展示命令与模型卡。导出先反量化合并为 `dist/bf16/`，再量化生成
`dist/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit/`。
明确加 `--upload` 才上传 HF，默认目标为 `myy555/LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit`；
其他贡献者通过 `--repo your-account/model-name` 指定自己的仓库，凭据由 HF 管理。

## 数据与许可

代码使用 [Apache-2.0](LICENSE)，基座和数据分别遵循来源许可。

| 来源 | 数据卡许可 / 使用方式 |
| --- | --- |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0，基座 |
| [unified-uncensored-qwen-chatml-sft](https://huggingface.co/datasets/usamakenway/unified-uncensored-qwen-chatml-sft) | 混合许可；默认仅取 Apache-2.0 的 Dolphin，OpenHermes / Airoboros 配额为 0 |
| [roleplay-zh-sharegpt-gpt4-data](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) | 数据卡 Apache-2.0，另有上游来源说明 |
| [deep-emotional-support-zh](https://huggingface.co/datasets/AngelWarmSmile123/deep-emotional-support-zh) | 数据卡 CC-BY-NC-SA-4.0 |

仓库公开代码与处理方法，不再分发原始对话。代码许可不代表混合数据或训练权重的许可；
使用与再分发时应核对来源条款，当前默认配方不能宣称为完全商业可用。

## 回归检查

使用标准库 `unittest`；Playwright 用于验证实际看板页面，开发依赖与训练依赖分开。

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m unittest discover -s tests -v
```

测试完全在临时目录构造微型 Qwen3 和 tokenizer，无需下载模型：

- 已有运行目录拒绝覆盖，包括失败的续训启动。
- 权重在新目录真实续载，旧目录中的所有文件保持不变。
- 8-bit 基座真实合并、导出，重新加载后的量化层与配置均为 4-bit。
- 模型卡准确标注记录范围，预览不写入导出文件。
- 浏览器显示全部保留数据来源与正确配比。

这些检查验证软件行为，不证明完整 1.7B 模型的量化质量、泛化能力或训练资源需求。
回归测试使用临时微型模型；完整训练与 HF 上传分别执行。
