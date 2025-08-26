# Kronos 项目分析文档

本文档旨在详细解释 Kronos 项目中的两个核心问题：如何对模型进行微调（Fine-tuning），以及如何使用其配套的分词器（Tokenizer）。

## 一、 如何微调模型 (Fine-tuning)

Kronos 项目提供了一个完整且清晰的微调流程，主要包含在 `finetune/` 目录下。整个过程被划分为四个主要步骤，让用户能够使用自己的数据来适配和优化预训练模型。

### 第 1 步：环境配置 (`finetune/config.py`)

这是微调前最关键的一步。您需要编辑 `finetune/config.py` 文件，来设置整个微调任务的参数。

**核心配置项包括：**

*   **路径设置 (Paths)**：您**必须**根据您本地的环境修改以下路径：
    *   `qlib_data_path`: 您本地存放 Qlib 数据的路径。
    *   `dataset_path`: 用于存放预处理后生成的数据集文件的目录。
    *   `save_path`: 用于保存微调过程中生成的模型检查点 (checkpoints) 的基础目录。
    *   `pretrained_tokenizer_path`: 您打算进行微调的**预训练分词器**的路径或其在 Hugging Face Hub 上的名称 (例如: `"NeoQuasar/Kronos-Tokenizer-base"`)。
    *   `pretrained_predictor_path`: 您打算进行微调的**预训练模型**的路径或其在 Hugging Face Hub 上的名称 (例如: `"NeoQuasar/Kronos-small"`)。
*   **数据参数 (Data Parameters)**：您可以定义数据的起止时间、回看周期 (`lookback_window`)、预测周期 (`predict_window`) 等。
*   **训练超参数 (Training Hyperparameters)**：您可以调整训练的 `epochs`、`batch_size`（批处理大小）、分词器和预测器的`learning_rate`（学习率）等。

### 第 2 步：数据准备 (`finetune/qlib_data_preprocess.py`)

配置完成后，运行数据预处理脚本。该脚本依赖 Qlib 库来高效处理金融时间序列数据。

```bash
python finetune/qlib_data_preprocess.py
```

该脚本会自动完成以下工作：
1.  从您在 `config.py` 中指定的路径加载原始数据。
2.  根据配置进行数据清洗、特征生成。
3.  将数据切分为训练集、验证集和测试集。
4.  将处理好的数据集保存为 `.pkl` 文件，存放在您指定的 `dataset_path` 目录下。

### 第 3 步：模型微调

模型微调分为两个阶段，分别是微调分词器和微调预测器。脚本设计为支持多 GPU 训练。

**3.1) 微调分词器 (`finetune/train_tokenizer.py`)**

这一步的目的是让预训练的分词器适应您自有数据的分布和特性。

```bash
# 将 NUM_GPUS 替换为您使用的 GPU 数量
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

训练完成后，最优的分词器模型将被保存在 `config.py` 中配置的路径下。

**3.2) 微调预测器 (`finetune/train_predictor.py`)**

在获得适应您数据的分词器后，用它来微调主模型（预测器）。

```bash
# 将 NUM_GPUS 替换为您使用的 GPU 数量
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

训练完成后，最优的预测器模型将被保存。

### 第 4 步：回测与评估 (`finetune/qlib_test.py`)

最后，您可以使用回测脚本来检验微调后模型在测试集上的实际表现。

```bash
# 指定用于推理的 GPU
python finetune/qlib_test.py --device cuda:0
```

该脚本会加载您微调好的模型，生成预测，并运行一个简单的策略回测，最终输出详细的性能报告和资金曲线图。

## 二、 如何使用分词器 (Tokenizer)

关于分词器的使用，项目设计得非常巧妙和用户友好。通常情况下，您**不需要直接操作分词器**。它的功能被完美地封装在了 `KronosPredictor` 这个类中。

`examples/prediction_example.py` 文件清晰地展示了标准的使用流程：

### 第 1 步：加载分词器和模型

首先，从 Hugging Face Hub 加载预训练的模型和其对应的分词器。

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 从 Hugging Face Hub 加载
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

### 第 2 步：实例化预测器

然后，将加载好的模型和分词器实例传递给 `KronosPredictor` 类。

```python
# 初始化预测器
predictor = KronosPredictor(model, tokenizer, device="cuda:0")
```

### 第 3 步：进行预测

最后，直接调用 `predictor` 实例的 `predict()` 方法，并传入您的原始数据（通常是 Pandas DataFrame 格式）。`KronosPredictor` 会在内部自动处理所有复杂的中间步骤，包括：

*   数据归一化。
*   **调用分词器**将连续的 K 线数据转换为模型能理解的离散化词元 (tokens)。
*   运行模型生成预测。
*   将模型的输出反-分词、反-归一化，最终返回给您一个格式干净、可直接使用的预测结果 DataFrame。

```python
# 假设 x_df, x_timestamp, y_timestamp 已经准备好
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len
)
```

总而言之，分词器是模型理解金融数据的关键“翻译官”，但它的使用被高度抽象和简化了。您只需要通过 `KronosPredictor` 这个高级接口，即可轻松完成从原始数据到最终预测的全过程。

---
希望这份文档对您有所帮助！
