# RepairLLaMA IR4×OR2 复现与改进实验

## 1. 项目简介

RepairLLaMA 是一个面向自动程序修复的代码大模型项目。原项目通过不同输入表示 IR 和输出表示 OR 组织训练数据，使大语言模型根据缺陷代码生成候选补丁。

本仓库复现并改进了 RepairLLaMA 中未公开权重的 `IR4×OR2` 组合。原项目公开了训练数据，但没有公开 IR4×OR2 CodeLLaMA 权重，因此我们基于公开数据重新训练，并在 AutoDL 平台上完成 LoRA、QLoRA 和动态 batch 推理实验。

## 2. 我们完成的工作

- 复现 RepairLLaMA 的 IR4×OR2 训练流程。
- 基于 CodeLLaMA-7B-fp16 训练 8bit LoRA 模型。
- 基于 CodeLLaMA-7B-fp16 训练 4bit QLoRA 模型。
- 实现 Defects4J 推理与 exact match 评测。
- 实现动态 batch 推理，根据输入长度调整 batch size。
- 对比 beam10、beam7、beam5 下的速度和准确率。
- 将训练得到的 LoRA / QLoRA adapter、推理结果和评测脚本整理为可复现实验仓库。

## 3. 改进方法

主要改进包括：

1. 使用 QLoRA 4bit 量化训练，降低训练阶段显存占用。
2. 在推理阶段实现动态 batch，根据输入长度设置不同 batch size，提高短输入样本吞吐。
3. 调整 beam size，在准确率和推理速度之间寻找折中方案。

其中，动态 batch 的策略是：

```text
短输入：batch size = 2
中等输入：batch size = 1
长输入：batch size = 1
```

这样可以在避免 CUDA OOM 的前提下提升部分样本的推理效率。

## 4. 实验环境

- 平台：AutoDL
- GPU：NVIDIA RTX 4090 24GB
- Python：3.10
- PyTorch：2.1.x
- transformers：4.40.2
- CUDA：12.x / 13.x 环境均测试过
- 主要依赖：datasets、peft、bitsandbytes、accelerate、sentencepiece、safetensors、tqdm

## 5. 实验结果

### 5.1 训练时间
| 配置       |   训练时长   | 
|---|---:|
| 8bit LoRA  |  约 11.704h |
| QLoRA 4bit |  约 9.895h  |

### 5.2 Defects4J 推理结果

| 实验方案 | 推理设置 | Defects4J exact hit | exact rate | 推理时间 |
|---|---:|---:|---:|---:|
| 8bit LoRA | beam10，逐条推理 | 107 / 439 | 24.37% | 约 75 分钟 |
| QLoRA 4bit | beam10，逐条推理 | 99 / 439 | 22.55% | 约 46 分钟 |
| QLoRA dynamic | beam5，short bs=2 | 86 / 438 | 19.63% | 20 分 59 秒 |
| 8bit LoRA dynamic | beam5，short bs=2 | 86 / 438 | 19.63% | 44 分 26 秒 |
| QLoRA dynamic | beam7，short bs=2 | 97 / 438 | 22.15% | 22 分 50 秒 |

### 5.3 训练显存对比

| 训练方式 | 峰值显存 |
|---|---:|
| 8bit LoRA | 约 12.0 GB |
| QLoRA 4bit | 约 8.4 GB |

QLoRA 相比 8bit LoRA 明显降低了训练显存占用。

## 6. 最终结论

8bit LoRA + beam10 的准确率最高，Defects4J exact rate 为 24.37%。

QLoRA dynamic beam7 bs2 是目前综合表现最好的方案：准确率为 22.15%，推理时间约 22 分 50 秒，相比普通 beam10 推理明显加速，同时保持了较高修复率。

因此，本项目最终推荐方案为：

```text
QLoRA 4bit + dynamic batch + beam7 + short_batch_size=2
```

该方案在训练显存、推理速度和修复准确率之间取得了较好的平衡。

## 7. 复现指南

### 7.1 克隆仓库

```bash
git clone git@github.com:xiaochahu-xd/repairllama-ir4or2-repro.git
cd repairllama-ir4or2-repro
```

也可以使用 HTTPS：

```bash
git clone https://github.com/xiaochahu-xd/repairllama-ir4or2-repro.git
cd repairllama-ir4or2-repro
```

### 7.2 安装依赖

```bash
pip install -r requirements.txt
```

如果需要手动安装，也可以执行：

```bash
pip install torch==2.1.2 transformers==4.40.2 datasets peft bitsandbytes accelerate sentencepiece safetensors tqdm
```

### 7.3 设置基座模型路径

本仓库不包含 CodeLLaMA-7B-fp16 基座模型，需要自行下载或提前缓存。

```bash
export BASE_MODEL_DIR=/path/to/CodeLLaMA-7B-fp16
```

AutoDL 中本实验使用的示例路径：

```bash
export BASE_MODEL_DIR=/root/autodl-tmp/hf-cache/models--TheBloke--CodeLlama-7B-fp16/snapshots/ce09049eb9140a19cf78051cb5d849607b6fa8ec
```

如果本地没有基座模型，可从 Hugging Face 下载：

```text
TheBloke/CodeLlama-7B-fp16
```

### 7.4 训练 8bit LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python src/lora/llama_sft_ir4or2_8bit.py \
  --output_dir outputs/repairllama-ir4xor2-8bit-lora-bs2 \
  --model_name_or_path $BASE_MODEL_DIR \
  --model_max_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --learning_rate 5e-4 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --logging_steps 1 \
  --fp16 true \
  --report_to none
```

训练完成后，adapter 默认保存在：

```text
outputs/repairllama-ir4xor2-8bit-lora-bs2
```

本仓库中已经保存了一份训练完成的 8bit LoRA adapter：

```text
adapters/repairllama-ir4xor2-8bit-lora-bs2
```

### 7.5 训练 QLoRA 4bit

```bash
CUDA_VISIBLE_DEVICES=0 python src/lora/llama_sft_ir4or2_qlora4bit.py \
  --output_dir outputs/repairllama-ir4xor2-qlora4bit-lora-bs2 \
  --model_name_or_path $BASE_MODEL_DIR \
  --model_max_length 1024 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --learning_rate 5e-4 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --logging_steps 1 \
  --fp16 true \
  --report_to none
```

训练完成后，adapter 默认保存在：

```text
outputs/repairllama-ir4xor2-qlora4bit-lora-bs2
```

本仓库中已经保存了一份训练完成的 QLoRA adapter：

```text
adapters/repairllama-ir4xor2-qlora4bit-lora-bs2
```

### 7.6 Defects4J 推理

推荐使用 QLoRA dynamic beam7 bs2 方案。

首先设置 adapter 路径：

```bash
export QLORA_DIR=adapters/repairllama-ir4xor2-qlora4bit-lora-bs2
```

然后运行推理：

```bash
CUDA_VISIBLE_DEVICES=0 python src/lora/llama_pred_ir4or2_qlora4bit_dynamic.py \
  --base_model_path $BASE_MODEL_DIR \
  --lora_path $QLORA_DIR \
  --data_path benchmarks/defects4j \
  --test_file defects4j_f2f_bugs_with_ir4.jsonl \
  --output_file results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_beams 7 \
  --request_num 7 \
  --short_batch_size 2 \
  --mid_batch_size 1 \
  --long_batch_size 1
```

如果要运行 8bit LoRA dynamic beam5 bs2：

```bash
export LORA_DIR=adapters/repairllama-ir4xor2-8bit-lora-bs2

CUDA_VISIBLE_DEVICES=0 python src/lora/llama_pred_ir4or2_8bit_dynamic.py \
  --base_model_path $BASE_MODEL_DIR \
  --lora_path $LORA_DIR \
  --data_path benchmarks/defects4j \
  --test_file defects4j_f2f_bugs_with_ir4.jsonl \
  --output_file results/preds/defects4j_ir4xor2_8bit_dynamic_beam5_bs2.jsonl \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_beams 5 \
  --request_num 5 \
  --short_batch_size 2 \
  --mid_batch_size 1 \
  --long_batch_size 1
```

### 7.7 计算准确率

整体 exact match：

```bash
python tools/eval_exact_ir4or2.py results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl
```

逐样本详细结果：

```bash
python tools/eval_exact_ir4or2_detail.py results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl
```

## 8. 文件说明

```text
src/lora/llama_sft_ir4or2_8bit.py
8bit LoRA 训练脚本。

src/lora/llama_sft_ir4or2_qlora4bit.py
QLoRA 4bit 训练脚本。

src/lora/llama_pred_ir4or2_8bit.py
8bit LoRA 普通推理脚本。

src/lora/llama_pred_ir4or2_qlora4bit.py
QLoRA 普通推理脚本。

src/lora/llama_pred_ir4or2_qlora4bit_dynamic.py
QLoRA 动态 batch 推理脚本。

src/lora/llama_pred_ir4or2_8bit_dynamic.py
8bit LoRA 动态 batch 推理脚本。

tools/eval_exact_ir4or2.py
整体 exact match 评测脚本。

tools/eval_exact_ir4or2_detail.py
逐样本详细评测脚本。

results/preds/
保存 Defects4J 推理结果。

adapters/
保存训练得到的 LoRA / QLoRA adapter。

requirements.txt
项目依赖文件。
```

## 9. 注意事项

1. 本仓库不包含 CodeLLaMA-7B-fp16 基座模型，需要自行下载或使用本地缓存。
2. 本仓库保存的是 LoRA / QLoRA adapter，不是完整合并后的大模型。
3. 推理时如果出现 `too long`，表示样本输入长度超过 `max_length=1024`，脚本会跳过该样本。
4. `num_beams` 越大，搜索更充分，准确率通常更高，但推理更慢、显存占用更高。
5. `request_num` 表示每个 bug 返回的候选补丁数量，不能大于 `num_beams`。
6. 动态 batch 是推理阶段优化，不影响训练得到的 adapter。
7. 如果出现 CUDA OOM，优先降低 `num_beams`、`request_num` 或 batch size。
8. 20 条样本只适合冒烟测试，最终结果应以完整 Defects4J 数据集为准。
9. `results/preds/*.jsonl` 中保存的是模型生成的候选补丁，需要通过 `tools/eval_exact_ir4or2.py` 或 `tools/eval_exact_ir4or2_detail.py` 计算准确率。
10. 若重新训练，请确保 Hugging Face 数据集 `ASSERT-KTH/repairllama-datasets` 可以访问，或已经在本地缓存。

## 10. 结果文件

本仓库保存了以下主要推理结果：

```text
results/preds/defects4j_ir4xor2_8bit_beam10.jsonl
8bit LoRA + beam10 普通推理结果。

results/preds/defects4j_ir4xor2_qlora4bit_beam10.jsonl
QLoRA 4bit + beam10 普通推理结果。

results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam5_bs2.jsonl
QLoRA 4bit + dynamic batch + beam5 + short bs=2 推理结果。

results/preds/defects4j_ir4xor2_8bit_dynamic_beam5_bs2.jsonl
8bit LoRA + dynamic batch + beam5 + short bs=2 推理结果。

results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl
QLoRA 4bit + dynamic batch + beam7 + short bs=2 推理结果。
```

## 11. 推荐运行方案

如果只想复现实验中推荐的最终结果，建议直接运行：

```bash
export BASE_MODEL_DIR=/path/to/CodeLLaMA-7B-fp16
export QLORA_DIR=adapters/repairllama-ir4xor2-qlora4bit-lora-bs2

CUDA_VISIBLE_DEVICES=0 python src/lora/llama_pred_ir4or2_qlora4bit_dynamic.py \
  --base_model_path $BASE_MODEL_DIR \
  --lora_path $QLORA_DIR \
  --data_path benchmarks/defects4j \
  --test_file defects4j_f2f_bugs_with_ir4.jsonl \
  --output_file results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_beams 7 \
  --request_num 7 \
  --short_batch_size 2 \
  --mid_batch_size 1 \
  --long_batch_size 1
```

然后计算准确率：

```bash
python tools/eval_exact_ir4or2.py results/preds/defects4j_ir4xor2_qlora4bit_dynamic_beam7_bs2.jsonl
```
