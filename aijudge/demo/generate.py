#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create Date: 2023/5/20 17:10
"""Description：本地推理接口
"""
import json

import fire
import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline


def generate(datas, tokenizer, model):
    """预测数据
    :param datas: json格式的数据
    :param tokenizer: tokenizer实例
    :param model: 模型实例
    :return:
    """
    def output_format(text):
        text = text.split("。本院认为，")[1].split("<生成结束>")[0]
        return text

    question_list = []
    for d_one in datas:
        question_list.append(d_one["input"].rstrip("。").rstrip("。本院认为，") + "。本院认为，")

    text_generator = TextGenerationPipeline(model, tokenizer, device=0)
    text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id
    generate_output_list = text_generator(question_list,
                                          max_length=1020,
                                          num_beams=1, top_p=0.8,
                                          num_return_sequences=1,
                                          eos_token_id=50256,
                                          pad_token_id=text_generator.model.config.eos_token_id)

    for d_one, g_one in zip(datas, generate_output_list):
        d_one["model_output"] = output_format(g_one[0]["generated_text"].replace(" ", ""))
    return datas


def main(
    base_model: str = "",
    input_file: str = "",
    output_file: str = "",
):
    """
    :param base_model: 模型地址
    :param input_file: 输入文件地址
    :param output_file: 输出文件地址
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model).to(device)

    with open(input_file, "r") as f:
        datas = json.load(f)

    generate_datas = generate(datas, tokenizer=tokenizer, model=model)

    with open(output_file, "w") as f:
        json.dump(generate_datas, f, indent=3, ensure_ascii=False)
    print("finish!!!")


if __name__ == "__main__":
    fire.Fire(main)