#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Create Date: 2023/4/14 21:41
"""Description：Web UI部署地址
"""

import gradio as gr
import fire
import torch

from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline


def main(
    model_path: str = "seussg/aijudge",
):
    """
    :param model_path: 模型地址
    :return:
    """
    def output_format(text):
        text = text.split("。本院认为，")[1].split("<生成结束>")[0]
        return text

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

    text_generator = TextGenerationPipeline(model, tokenizer, device=0)
    text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id

    model.eval()

    def evaluate(
        instruction="本院认为",
        input=None
    ):
        question = input.rstrip("。").rstrip("。本院认为，") + "。本院认为，"
        with torch.no_grad():
            predict_output = text_generator(question,
                                            max_length=1020,
                                            top_p=0.7,
                                            do_sample=True,
                                            num_return_sequences=5,
                                            eos_token_id=50256,
                                            pad_token_id=text_generator.model.config.eos_token_id)

        output = ""
        for idx, o_one in enumerate(predict_output):
            output += f"结果【{idx+1}】:\n" + output_format(o_one["generated_text"].replace(" ", "")) + "\n\n"
        return output

    gr.Interface(
        fn=evaluate,
        theme="grass",
        inputs=[
            gr.components.Textbox(lines=2, label="指令", placeholder="本院认为", interactive=False),
            gr.components.Textbox(lines=10, label="输入", placeholder="指控"),
        ],
        outputs=[
            gr.inputs.Textbox(lines=25, label="预测结果")
        ],
        flagging_options=["保存"],
        title="AI Judge",
        description="AI Judge：基于法院观点生成的可解释法律判决预测模型",

    ).launch(share=True)


if __name__ == "__main__":
    fire.Fire(main)
