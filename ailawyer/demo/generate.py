#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create Date: 2023/5/20 17:10
"""Description：本地推理接口
"""
import json

import fire
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

prompt_prefix = "假设你是一名律师，请分析如下案例，并提供专业的法律服务。"


def generate(datas, tokenizer, model):
    """预测数据
    :param datas: json格式的数据
    :param tokenizer: tokenizer实例
    :param model: 模型实例
    :return:
    """
    p_bar = tqdm(datas)
    for d_one in p_bar:
        prompt = generate_prompt(prompt_prefix, d_one["input"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(temperature=0.1, top_p=0.8, top_k=1, num_beams=1)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=625,
            )
        output_ids = generation_output.sequences[0]
        output = tokenizer.decode(output_ids)
        answer = output.split("### Response:")[1].strip().strip("</s>")
        d_one["model_output"] = answer
        p_bar.set_description("Processing")
    return datas


def main(
    base_model: str = "",
    lora_weights: str = "",
    input_file: str = "",
    output_file: str = "",
    load_8bit: bool = False,
    cache_dir: str = ""
):
    """
    :param base_model: base模型权重
    :param lora_weights: 本项目lora权重
    :param input_file: 输入文件地址
    :param output_file: 输出文件地址
    :param load_8bit:
    :return:
    """
    tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
    model = LlamaForCausalLM.from_pretrained(base_model,
                                             load_in_8bit=False,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16).half()

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    with open(input_file, "r") as f:
        datas = json.load(f)

    generate_datas = generate(datas, tokenizer=tokenizer, model=model)

    with open(output_file, "w") as f:
        json.dump(generate_datas, f, indent=3, ensure_ascii=False)
    print("finish!!!")


if __name__ == "__main__":
    fire.Fire(main)