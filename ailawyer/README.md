## 环境准备
- 机器：内存30G+、GPU显存32G+
- python环境：3.9以上
- 依赖包：安装项目requirements.txt安装即可

## 模型合并

### Step 1：下载原版LLaMa 13B模型参数
包含文件如下：
- consolidated.*.pth
- tokenizer.model
- params.json

注：因要求无法提供下载地址，请自行下载

### Step 2：下载Chinese-LLaMA-Alpaca的13B权重
- HF地址：https://huggingface.co/ziqingyang/chinese-llama-lora-13b/tree/main 可以通过huggingface的接口下载或者手动保存地址下载
- 网盘地址：https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t

保存的文件夹下需包含adapter_config.json、adapter_model.bin、special_tokens_map.json、tokenizer.model、tokenizer_config.json

### Step 3：将原版LLaMa转化HF格式参数
使用代码utils/convert_llama_weights_to_hf.py进行格式转化。
```python
python utils/convert_llama_weights_to_hf.py \
    --input_dir origin_llama_weight_directory \
    --model_size 13B \
    --output_dir origin_llama_hf_weight_directory
```
参数如下：

- input_dir：原版LLaMa参数文件目录，目录下放tokenizer.model，同时新建一个名为 13B 的目录，并将原版参数中其他文件放在{input_dir}/13B 下面，具体的{input_dir}文件夹结构如下
```text
--tokenizer.model
--13B
    --consolidated.*.pth
    --params.json
```
- output_dir：保存转化后参数的目录

### Step 4：合并LoRA权重生成base模型全量权重

合并扩充Chinese-LLaMA-Alpaca的13B的权重，主要是扩充词表和合并LoRA权重，合并后的模型作为本项目训练和部署的base模型

```python
python utils/merge_llama_with_chinese_lora.py \
    --base_model origin_llama_hf_weight_directory \
    --lora_model chinese_llama_alpaca_lora_weight_directory \
    --output_type huggingface \
    --output_dir save_merge_weight_directory
```
参数如下：

- base_model：存放HF格式的LLaMA模型权重和配置文件的目录（Step 3保存的目录）
- lora_model：Chinese-LLaMA-Alpaca（Step 2）的权重目录
- output_type: 指定输出格式，huggingface
- output_dir：指定保存全量模型权重的目录
- offload_dir：可选，对于低内存用户需要指定一个offload缓存路径


## 本地推理&部署

本项目需要在GPU上运行

### 下载本项目LoRA权重

下载地址：https://huggingface.co/seussg/ailawyer/tree/main

### Web UI部署

采用Gradio Web UI进行本地部署，在0号GPU上部署如下

```shell
CUDA_VISIBLE_DEVICES=0 python demo/web_demo_llama_13B.py \
    --base_model base_model_weight \
    --lora_weights ai_lawer_lora_weight
```
参数如下：

- base_model 模型合并后的全量权重目录
- lora_weights 本项目的LoRA权重

部署成功后，获取url即可体验

### 本地推理

给定文件数据进行生成，使用0号GPU进行生成如下

```shell
CUDA_VISIBLE_DEVICES=0 python demo/generate.py \
    --base_model base_model_weight \
    --lora_weights ai_lawer_lora_weight \
    --input_file data/samples.json \
    --output_file save_file_path
```
参数如下：
- base_model 模型合并后的全量权重目录
- lora_weights 本项目的LoRA权重
- input_file 输入文件，格式参考data/samples.json
- output_file 生成结果后保存的文件，生成内容保存在model_output字段中












