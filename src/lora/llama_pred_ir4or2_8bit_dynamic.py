import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)


@dataclass
class ModelArguments:
    base_model_path: Optional[str] = field(default=None)
    lora_path: Optional[str] = field(default=None)
    max_length: int = field(default=1024)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    test_file: str = field(default=None)
    output_file: str = field(default=None)


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=256)
    num_beams: int = field(default=10)
    request_num: int = field(default=10)
    short_batch_size: int = field(default=4)
    mid_batch_size: int = field(default=2)
    long_batch_size: int = field(default=1)
    short_len: int = field(default=512)
    mid_len: int = field(default=768)


def choose_batch_size(max_len, args):
    if max_len <= args.short_len:
        return args.short_batch_size
    if max_len <= args.mid_len:
        return args.mid_batch_size
    return args.long_batch_size


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        ),
    )

    model = PeftModel.from_pretrained(
        model,
        model_args.lora_path,
        torch_dtype=torch.float16,
    )
    model.eval()

    model.config.pad_token_id = tokenizer.pad_token_id

    data_file = Path(data_args.data_path) / data_args.test_file
    samples = [json.loads(line) for line in open(data_file, encoding="utf-8")]

    prepared = []
    skipped = 0

    for sample in samples:
        buggy_code = sample["buggy_code"]
        tokenized = tokenizer(buggy_code, return_tensors=None)
        input_len = len(tokenized["input_ids"])

        if input_len > model_args.max_length:
            print(f"The code sequence of bug {sample['bug_id']} is too long, {input_len}.")
            skipped += 1
            continue

        prepared.append({
            "sample": sample,
            "input_len": input_len,
        })

    prepared.sort(key=lambda x: x["input_len"])

    output_path = Path(data_args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    i = 0

    pbar = tqdm(total=len(prepared), desc="Generating dynamic batch...")

    while i < len(prepared):
        current_len = prepared[i]["input_len"]
        bs = choose_batch_size(current_len, gen_args)
        batch_items = prepared[i:i + bs]

        texts = [x["sample"]["buggy_code"] for x in batch_items]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_args.max_length,
        ).to(model.device)

        prompt_len = inputs.input_ids.shape[1]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_args.max_new_tokens,
                    num_beams=gen_args.num_beams,
                    num_return_sequences=gen_args.request_num,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                print(f"OOM on single sample: {batch_items[0]['sample']['bug_id']}")
                i += 1
                pbar.update(1)
                continue
            bs = 1
            batch_items = prepared[i:i + bs]
            texts = [x["sample"]["buggy_code"] for x in batch_items]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model_args.max_length,
            ).to(model.device)
            prompt_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_args.max_new_tokens,
                    num_beams=gen_args.num_beams,
                    num_return_sequences=gen_args.request_num,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        output_ids = outputs[:, prompt_len:]
        decoded_patches = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        full_outputs = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for b, item in enumerate(batch_items):
            sample = item["sample"]
            output_dict = {}

            start = b * gen_args.request_num
            end = start + gen_args.request_num

            for j, (full_text, patch_text) in enumerate(
                zip(full_outputs[start:end], decoded_patches[start:end])
            ):
                output_dict[j] = {
                    "original_output": full_text,
                    "output_patch": patch_text,
                }

            results.append({
                "bug_id": sample["bug_id"],
                "output": output_dict,
                "buggy_code": sample["buggy_code"],
                "gold_patch": sample.get("fixed_chunk", ""),
            })

        i += len(batch_items)
        pbar.update(len(batch_items))

        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

    pbar.close()
    print(f"Done. generated={len(results)}, skipped={skipped}, output={output_path}")


if __name__ == "__main__":
    main()
