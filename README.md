# RepairLLaMA IR4×OR2 复现与改进实验

## 1. 项目简介

RepairLLaMA 是一个面向自动程序修复的代码大模型项目。原项目通过不同输入表示 IR 和输出表示 OR 组织训练数据，使大语言模型根据缺陷代码生成补丁。

本仓库复现并改进了 RepairLLaMA 中未公开权重的 `IR4×OR2` 组合。原项目公开了训练数据，但没有公开 IR4×OR2 CodeLLaMA 权重，因此我们基于公开数据重新训练，并在 AutoDL 平台上完成 LoRA、QLoRA 和动态 batch 推理实验。

## 2. 我们完成的工作

- 复现 RepairLLaMA 的 IR4×OR2 训练流程。
- 基于 CodeLLaMA-7B-fp16 训练 8bit LoRA 模型。
- 基于 CodeLLaMA-7B-fp16 训练 4bit QLoRA 模型。
- 实现 Defects4J 推理与 exact match 评测。
- 实现动态 batch 推理，根据输入长度调整 batch size。
- 对比 beam10、beam7、beam5 下的速度和准确率。

## 3. 改进方法

主要改进包括：

1. 使用 QLoRA 4bit 量化训练，降低训练显存。
2. 使用动态 batch 推理，提高短输入样本的推理吞吐。
3. 调整 beam size，在准确率和推理速度之间寻找折中方案。

## 4. 实验环境

- 平台：AutoDL
- GPU：RTX 4090 24GB
- Python：3.10
- PyTorch：2.1.x
- transformers：4.40.2
- CUDA：12.x / 13.x
- 主要依赖：datasets、peft、bitsandbytes、accelerate、sentencepiece

## 5. 实验结果

| 实验方案 | 推理设置 | Defects4J exact hit | exact rate | 推理时间 |
|---|---:|---:|---:|---:|
| 8bit LoRA | beam10，逐条推理 | 107 / 439 | 24.37% | 约 46 分钟 |
| QLoRA 4bit | beam10，逐条推理 | 99 / 439 | 22.55% | 约 46 分钟 |
| QLoRA dynamic | beam5，short bs=2 | 86 / 438 | 19.63% | 20 分 59 秒 |
| 8bit LoRA dynamic | beam5，short bs=2 | 86 / 438 | 19.63% | 44 分 26 秒 |
| QLoRA dynamic | beam7，short bs=2 | 97 / 438 | 22.15% | 22 分 50 秒 |

训练显存对比：

| 训练方式 | 峰值显存 |
|---|---:|
| 8bit LoRA | 约 12.0 GB |
| QLoRA 4bit | 约 8.4 GB |

## 6. 最终结论

8bit LoRA + beam10 的准确率最高，Defects4J exact rate 为 24.37%。

QLoRA dynamic beam7 bs2 是目前综合表现最好的方案：准确率为 22.15%，推理时间约 22 分 50 秒，相比普通 beam10 推理明显加速，同时保持了较高修复率。

因此，本项目最终推荐方案为：

```text
QLoRA 4bit + dynamic batch + beam7 + short_batch_size=2



