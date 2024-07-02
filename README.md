# 代码生成Prompt的自动重构
这个仓库是PromptReform的代码仓库，分为以下三个部分：
- code_generation: 包含两种大模型的代码生成脚本代码；
- prompt_reformulation: 包含对prompt测试集重构的代码；
- t5_pretraining:包含t5模型预训练数据和代码。
## PromptReform
PromptReform是一种基于自监督学习的prompt自动重构方法。它的核心优势在于无需依赖昂贵的<原始prompt，重构后prompt>平行语料库，而是有效利用广泛存在的单向代码注释数据，通过自监督的掩码跨度预测任务对模型进行预训练，构建prompt重构模型。PromptReform方法以T5为主干模型，包括三个关键步骤：
- 首先，通过从大量代码注释中抽取的自然语言描述，对T5模型进行自监督的完形填空式任务训练。通过持续预训练，让模型深入学习和理解代码生成场景中常见的prompt结构、语法和语义内涵。
- 接着，使用训练后的T5模型来预测prompt内部可能存在的信息缺失位置，并补充缺失信息。
- 最后将重构后的prompt作为输入，让大模型生成更准确的代码。
