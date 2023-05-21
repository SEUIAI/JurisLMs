## 环境准备
- 机器：GPU显存16G+
- python环境：3.9以上
- 依赖包：安装项目requirements.txt安装即可

## 本地推理&部署

### Web UI部署

采用Gradio Web UI进行本地部署，在0号GPU上部署如下

```shell
CUDA_VISIBLE_DEVICES=0 python demo/web_demo_gpt2.py \
    --base_model seussg/aijudge
```
参数如下：

- base_model 本项目的huggingface模型

部署成功后，获取url即可体验

### 本地推理

给定文件数据进行生成，使用0号GPU进行生成如下

```shell
CUDA_VISIBLE_DEVICES=0 python demo/generate.py \
    --base_model base_model_weight \
    --input_file data/samples.json \
    --output_file save_file_path
```
参数如下：
- base_model 本项目模型
- input_file 输入文件，格式参考data/samples.json
- output_file 生成结果后保存的文件，生成内容保存在model_output字段中

