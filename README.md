## Dia-Pangu
基于盘古大模型多模态医学影像与文本增量与推理项目
### 模型结构

![Project-structure-diagram](Fig/project-structure-diagram.png)

本模型的LLM部分使用开源的openpangu_7b，Diagnostic Text Generation部分采用[Dia-LLaMA](https://github.com/zhi-xuan-chen/Dia-LLaMA)开源的模型，数据集采用开源数据集[CTRG](https://github.com/tangyuhao2016/CTRG)中的CTRG-Chest-548K部分。本项目主要以openpangu_7b为核心，融合Dia-LLaMA的方法从而构建基于盘古大模型多模态医学影像与文本增量与推理模型。
 ### 代码结构
 #### 模型的核心实现代码放在/dia-pangu/src文件夹中,核心部分由以下文件构成：
 train_pre.py为pangu视觉Perceiver训练入口，负责解析命令行参数（模型路径、训练参数等），加载数据集，初始化模型、分词器和Trainer，并启动整个训练流程。

 train_ft.py为pangu微调训练入口，与train_pre.py类似
 
 test.py为模型验证入口，负责加载训练好的权重，并对验证集进行验证，并将结果保存在/dia-pangu/results/dia-pangu.csv中。
 
 /dia-pangu/src/Dataset中存放训练和验证所用的中文dataloader，负责将CTRG-Chest-548K数据集的中文版本加载到模型。

 /dia-pangu/src/Dataset-EN中存放训练和验证所用的英文dataloader，将CTRG-Chest-548K数据集的英文部分加载到模型，大部分代码与Dataset文件中一致，不过为了方便使用，代码中已经对标签和路径做了区分，使用时注意更改为本地路径。
 
 /dia-pangu/src/My_Trainer中存放自定义的训练器trainer.py，实现了完整的训练和评估循环（Training/Evaluation Loop）。它管理模型的正向/反向传播、优化器（Optimizer）的更新、混合精度训练（BF16）、学习率调度、以及模型的保存与加载。
 
 /dia-pangu/src/Model中存放模型各模块的实现代码，其中multimodality_model.py封装了主模型，定义了 MultiPanguForCausalLM 类，整合了 OpenPangu-7B 语言模型和 MyEmbedding 视觉模块，并且定义了最终的组合损失函数，将语言模型损失和对比损失结合起来。
 
 My_embedding.py为多模态的核心，定义了 MyEmbedding 类，负责处理所有视觉输入和多模态逻辑。它包含了：1. 3D Vision Encoder (ViT) 2. Perceiver Resampler 和 Linear Projection 层 3. 完整的对比学习实现，包括疾病原型、正负例记忆库以及相似度计算等。
 
 #### 训练与验证的脚本配置文件放在/dia-pangu/scripts文件夹中：
 train.sh为训练脚本，其中配置了数据集路径、模型路径、权重保存路径、epoch数、学习率以及训练节点数等常规配置。
 test.sh为验证脚本，其中仅配置了验证入口。
 ### 训练复现
 #### 环境及其他准备
 本模型训练所需环境已写入requirements.txt，请于/dia-pangu目录下运行
```bash
# create environment
conda create --name dia python=3.11
# install requirements
pip install -r requirements
conda activate dia
```
另外请准备openpangu-7b的模型、原Dia-LLaMA的权重及CTRG-Chest-548K数据集，模型可从[openpangu-7B](https://atomgit.com/ascend-tribe/openpangu-embedded-7b-model)下载，权重可从[Dia-LLaMA](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)下载，并将其放在/dia-pangu/checkpoint目录下，数据集可从[CTRG-Chest-548K](https://huggingface.co/datasets/Trusure/CTRG-Chest-548K_volume)下载。
 #### 训练
请于/dia-pangu目录下先运行：
```bash
sh scripts/train_pre.sh
```
以确保pangu的视觉Perceiver可以适应CT数据，确保模型多模态能力
请注意调整/dia-pangu/src/train.py中的模型路径及/dia-pangu/src/Dataset/dataset_train.py中的数据集路径，训练的参数可以进行自定义，训练后的权重放在/dia-pangu/checkpoint_new目录下。

随后可在相同目录下运行
```bash
sh scripts/train_ft.sh
```
以对pangu进行微调，目前使用Lora方法进行微调，参数可以在sh命令行内自定义
 #### 推理
请于/dia-pangu目录下运行：
```bash
sh scripts/test.sh
```

在训练和推理过程中，请注意调整test.py/train_ft.py/train_pre.py中的模型路径并且中英文的根据需要改变数据集的头文件调用:
```bash
from Dataset.dataset_test import dataset_test
```
这里如使用中文则调用Dataset，使用英文则调用Dataset-EN，注意修改dataset_test.py/dataset_train.py中的数据集路径为本地路径。
 
## Dia-Pangu-pruning
以基于盘古大模型多模态医学影像与文本增量与推理项目作为验证指标的pangu剪枝技术研究，对pangu大语言模型尝试了不同剪枝率的剪枝
### 代码结构
剪枝部分包含敏感性分析，实际剪枝工具以及剪枝后测试工具
#### 敏感性分析工具代码位于/dia-pangu-pruning/sensitivity-analyse中，核心代码包括：
layer_mask.py为层屏蔽工具，通过将特定投影层（Q/K/V）的输出置零，动态屏蔽 Transformer 各注意力层。

layer_sensitivity_scan.py为实际敏感性分析工具，功能如下：
1.建立基线：在未修改模型的情况下，记录所有测试提示词的输出 logits

2.逐层屏蔽：依次屏蔽每一层的注意力机制（Q/V 或仅 V）

3.计算偏差：使用 log-probability distance 度量屏蔽前后输出的差异

4.生成报告-csv格式结果：输出每层的平均敏感度分数和标准差
