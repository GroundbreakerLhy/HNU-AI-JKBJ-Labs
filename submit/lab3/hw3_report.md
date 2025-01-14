<center><font face="黑体" size=7><b>湖南大学信息科学与工程学院</center>
<center><font face="宋体" size=6>人工智能 课程实验报告</center>
<br>
<font face="宋体" size=4>指导老师：<u>张子兴老师</u> 实验日期：<u>2024</u>年<u>12</u>月<u>30</u>日
<br>
<font face="宋体" size=4>实验项目名称：<u>文本单模态深度学习</u>
<hr>

## 一、	实验背景与目标

本实验任务是通过文本数据进行情感识别任务，实现文本单模态深度学习。

## 二、	实验方法

### 2.1 数据处理

#### 2.1.1 EDA (Easy Data Augmentation)
EDA (Easy Data Augmentation)：用于提高文本分类任务性能的简单数据增强技术。EDA 包含四个简单但功能强大的操作：同义词替换、随机插入、随机交换和随机删除。^[(Wei, Jason, and Kai Zou. "Eda: Easy data augmentation techniques for boosting performance on text classification tasks." arXiv preprint arXiv:1901.11196 (2019).)]

#### 2.1.2 MLM (Masked Language Model)
Masked Language Model：用于文本分类任务的预训练模型。MLM 通过预训练模型`Bert`，可以提取文本的语义信息，提高文本分类任务的性能。

#### 2.1.3 [Augment.py](/submit/lab3/augment.py)
```python
def synonym_replacement(words, n):
    # ...
    return new_words
def get_synonyms(word):
    # ...
    return list(synonyms)
def random_insertion(words, n):
    # ...
    return new_words
def add_word(new_words):
    # ...
    return new_words
def random_swap(words, n):
    # ...
    return new_words
def swap_word(new_words):
    # ...
    return new_words
def random_deletion(words, p):
    # ...
    return new_words
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    # ...
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(" ".join(a_words))

    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(" ".join(a_words))

    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(" ".join(a_words))
    # ...
    return augmented_sentences

class LLMAugmenter:
    def __init__(self, model_name="bert-base-uncased"):
        # ...

    def _get_prompt_template(self, text, emotion):
        # ...
        return random.choice(templates)

    def generate(self, text, emotion, num_samples=1, temperature=0.7, max_length=128):
        # ...
        return generated_texts

    def augment_dataset(self, texts, emotions, samples_per_text=1):
        # ...
        return augmented_texts, augmented_emotions
```

#### 2.1.4 其他高质量数据集 (EmotionLines and Emotions dataset for NLP)

`EmotionLines` dataset contains labeled emotion on every utterance in dialogues from Friends TV scripts and EmotionPush chat logs.^[Chen, Sheng-Yeh, et al. "Emotionlines: An emotion corpus of multi-party conversations." arXiv preprint arXiv:1802.08379 (2018).]

`Emotions dataset for NLP` is a collection of documents and its emotions, It helps greatly in NLP Classification tasks^[https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp]

### 2.2 模型设计

#### 2.2.1 模型架构

**Llama**:
Llama 3.2 多语种大型语言模型（LLMs）集合是一个经过预训练和指令调整的生成模型集合，有 1B 和 3B 两种大小（文本输入/文本输出）。 Llama 3.2 经指令调整的纯文本模型针对多语言对话用例进行了优化，包括代理检索和摘要任务。 在常见的行业基准测试中，它们的表现优于许多现有的开源和封闭聊天模型。^[https://huggingface.co/meta-Llama/Llama-3.2-1B]

本实验基于 Llama 模型进行序列分类任务设计,主要包含以下组件:

1. **基础模型**: 使用 Llama-3.2-1B 作为 backbone
2. **分类头**: 在 Llama 基础上添加序列分类层,输出4种情感类别
3. **标签平滑**: 使用 `LabelSmoothing` 提高模型鲁棒性

#### 2.2.2 关键超参数

- `batch_size`: 24
- `learning_rate`: 8e-6
- `weight_decay`: 0.1
- `smoothing`: 0.1
- `patience`: 3
- `epoch`: 20
- `max_length`: 512

#### 2.2.3 优化策略

1. **优化器**: `AdamW (eps=1e-8, betas=(0.9, 0.999))`
2. **早停机制**: 验证集F1分数无提升超过8轮则停止训练
3. **Prompt工程**: 添加任务相关提示增强模型理解
4. **数据增强**: 使用 [EDA](#211-eda-easy-data-augmentation) 和 [MLM](#212-masked-language-model) 增强数据集

#### 2.2.4 损失函数设计

标签平滑的交叉熵损失函数是一种用于改善模型泛化性能的技术。通过将真实标签的概率分布平滑化，能够减少模型对噪声和过拟合的敏感性。其公式如下:

$$
L_{smooth} = -\sum_{c=1}^{C}\Bigl[\bigl(1 - \alpha\bigr)\delta_{c,y} \;+\; \frac{\alpha}{C}\Bigr]\log(p_c)
$$

实现标签平滑的交叉熵损失:
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss.squeeze(1) + self.smoothing * smooth_loss
        return loss.mean()
```

### 2.3 性能评估

本实验采准确率(Accuracy)、不平衡精度(UA)、F1 值、精确率(Precision)和混淆矩阵来评估模型的性能。

1. **准确率(Accuracy)**: $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **不平衡精度(UA)**: $$UA = \frac{1}{2} \cdot \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$
3. **F1 值**: $$F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$
4. **精确率(Precision)**: $$Precision = \frac{TP}{TP + FP}$$
5. **混淆矩阵**: 混淆矩阵是一种用于可视化分类模型性能的矩阵，它展示了模型在不同类别上的预测结果。

```python
def calculate_score_classification(preds, labels, average_f1="macro"):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    ua = recall_score(labels, preds, average="macro", zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix
```

## 三、	实验设置和模型训练

### 3.1 实验环境

- **硬件环境**: NVIDIA  A100 80GB PCIe, Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
- **软件环境**: Python 3.8, PyTorch 2.0.1, Transformers 4.46.3

### 3.2 模型训练

#### 3.2.1 训练过程设计

1. **数据流设计**:
   - 输入文本通过`tokenizer`编码为`input_ids`和`attention_mask`
   - 使用`DataLoader`进行批处理,训练时随机采样
   - 验证和测试时使用顺序采样保持结果一致性

2. **训练循环**:
   - 每轮遍历训练集进行参数更新
   - 计算平均训练损失
   - 在验证集上评估F1分数
   - 保存最佳模型权重
   - 检查早停条件

3. **评估指标**:
   - 主要指标: `Macro F1`分数
   - 辅助指标: `Accuracy`和验证损失
   - 早停判断: 验证集F1分数

4. **模型预测**:
   - 加载最佳权重状态
   - 对测试集数据进行批量推理
   - 输出类别预测结果
   - 保存预测结果到CSV文件

#### 3.2.2 关键代码实现

**模型训练核心代码**:
```python
def train(self, dataloader_train, dataloader_dev, epochs):
    for epoch in range(epochs):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        avg_train_loss = total_loss / len(dataloader_train)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        val_f1 = self.evaluate(dataloader_dev)

        if val_f1 > self.best_f1_score:
            self.best_f1_score = val_f1
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), "best_model_hw3.pt")
            print("Validation F1 Score improved, saving model.\n")
        else:
            self.early_stop_counter += 1
            print(
                f"No improvement in validation F1 Score for {self.early_stop_counter} epoch(s).\n"
            )
            if self.early_stop_counter >= self.patience:
                print(
                    f"Early stopping triggered at F1 Score: {self.best_f1_score:.4f}"
                )
                break
```


## 四、	实验结果与分析

```bash
/home/mengquan/miniconda3/envs/lhy/lib/python3.8/site-packages/torchvision/datapoints/__init__.py:12: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
/home/mengquan/miniconda3/envs/lhy/lib/python3.8/site-packages/torchvision/transforms/v2/__init__.py:54: UserWarning: The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta. While we do not expect major breaking changes, some APIs may still change according to user feedback. Please submit any feedback you may have in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check out https://github.com/pytorch/vision/issues/7319 to learn more about the APIs that we suspect might involve future changes. You can silence this warning by calling torchvision.disable_beta_transforms_warning().
  warnings.warn(_BETA_TRANSFORMS_WARNING)
Some weights of LlamaForSequenceClassification were not initialized from the model checkpoint at ./Llama-3.2-1B and are newly initialized: ['score.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch 1/20: 100%|█████████| 544/544 [51:50<00:00,  5.72s/it]
Epoch 1/20, Training Loss: 1.2611
Validation Loss: 1.1979, Accuracy: 0.5626, F1 Score: 0.5576
Validation F1 Score improved, saving model.

Epoch 2/20: 100%|█████████| 544/544 [51:51<00:00,  5.74s/it]
Epoch 2/20, Training Loss: 1.0701
Validation Loss: 1.0862, Accuracy: 0.6382, F1 Score: 0.6326
Validation F1 Score improved, saving model.

Epoch 3/20: 100%|█████████| 544/544 [51:50<00:00,  5.72s/it]
Epoch 3/20, Training Loss: 0.9596
Validation Loss: 1.1129, Accuracy: 0.6353, F1 Score: 0.6275
No improvement in validation F1 Score for 1 epoch(s). Best F1: 0.6326

Epoch 4/20: 100%|█████████| 544/544 [51:51<00:00,  5.73s/it]
Epoch 4/20, Training Loss: 0.7758
Validation Loss: 1.1427, Accuracy: 0.6537, F1 Score: 0.6432
Validation F1 Score improved, saving model.

Epoch 5/20: 100%|█████████| 544/544 [51:50<00:00,  5.72s/it]
Epoch 5/20, Training Loss: 0.7185
Validation Loss: 1.1666, Accuracy: 0.6460, F1 Score: 0.6413
No improvement in validation F1 Score for 1 epoch(s). Best F1: 0.6432

Epoch 6/20: 100%|█████████| 544/544 [51:50<00:00,  5.72s/it]
Epoch 1/20, Training Loss: 0.6944
Validation Loss: 0.9574, Accuracy: 0.7177, F1 Score: 0.7128
Validation F1 Score improved, saving model.

Epoch 7/20: 100%|█████████| 544/544 [51:48<00:00,  5.71s/it]
Epoch 2/20, Training Loss: 0.4877
Validation Loss: 1.0022, Accuracy: 0.6945, F1 Score: 0.6918
No improvement in validation F1 Score for 1 epoch(s). Best F1: 0.7128

Epoch 8/20: 100%|█████████| 544/544 [51:48<00:00,  5.71/it]
Epoch 3/20, Training Loss: 0.4507
Validation Loss: 1.0734, Accuracy: 0.6751, F1 Score: 0.6736
No improvement in validation F1 Score for 2 epoch(s). Best F1: 0.7128
Early stopping triggered at F1 Score: 0.6629

Epoch 9/20: 100%|█████████| 544/544 [51:50<00:00,  5.72s/it]
Epoch 9/20, Training Loss: 0.4718
Validation Loss: 1.2446, Accuracy: 0.6625, F1 Score: 0.6572
Early stopping triggered at F1 Score: 0.7128
测试集预测结果已成功写入到文件中!
```

## 五、 总结与展望

本实验通过使用 Llama 模型进行情感分类任务，实现了文本单模态深度学习。通过数据增强技术和预训练模型，提高了模型的性能。

在未来，可以尝试更多的数据增强技术和模型结构，提高模型的性能和泛化能力。
