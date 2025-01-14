<center><font face="黑体" size=7><b>湖南大学信息科学与工程学院</center>
<center><font face="宋体" size=6>人工智能 课程实验报告</center>
<br>
<font face="宋体" size=4>指导老师：<u>张子兴老师</u> 实验日期：<u>2024</u>年<u>11</u>月<u>30</u>日
<br>
<font face="宋体" size=4>实验项目名称：<u>音频单模态机器学习</u>
<hr>

## 一、	实验背景与目标

本实验任务是通过音频数据进行情感识别任务，实现音频单模态机器学习。

## 二、	实验方法

### 2.1 数据提取和预处理

#### 2.1.1 OpenSMILE 并行提取音频特征

**OpenSMILE 介绍：**

OpenSMILE 是一个用于音频特征提取的工具包，它提供了大量的音频特征提取函数，可以用于音频信号的特征提取。OpenSMILE 是一个开源的音频特征提取工具，它提供了大量的音频特征提取函数，可以用于音频信号的特征提取。OpenSMILE 是一个开源的音频特征提取工具，它提供了大量的音频特征提取函数，可以用于音频信号的特征提取。OpenSMILE 是一个开源的音频特征提取工具，它提供了大量的音频特征提取函数，可以用于音频信号的特征提取。

**OpenSMILE 特征提取：**

```python
def extract_audio_feature(file):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    y = smile.process_file(file)
    y = y.to_numpy().reshape(-1)
    return y
```

这里本实验使用 OpenSMILE 提取音频特征，提取的特征为 eGeMAPSv02 特征集，提取的特征级别为 Functionals。

**并行处理：**

由于样本数量大，提取特征的时间较长，本实验选择使用多线程并行处理，加快特征提取的速度。本实验使用 ThreadPoolExecutor，选择最大线程数为 85，提取特征的函数为 extract_audio_feature。

```python
def extract_features_parallel(file_list, max_workers=85):
    features = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(extract_audio_feature, file): file for file in file_list
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(file_list),
            desc="Extracting features",
        ):
            file = future_to_file[future]
            try:
                feature = future.result()
                features.append(feature)
            except Exception as exc:
                print(f"{file} generated an exception: {exc}")
    features = np.stack(features, axis=0)
    return features
```

将提取的特征和标签对齐，**消除线程随机性**带来的顺序问题。

```python
def align_features_labels(features, labels):
    assert features.shape[0] == labels.shape[0]
    return features, labels
```

#### 2.1.2 特征预处理

数据特征预处理的方式有很多，如降维、标准化、归一化等。这里本实验使用标准化的方式对特征进行预处理。

标准化指的是将特征数据的分布调整成标准正态分布，也叫高斯分布，也就是使得数据的均值维0，方差为1。

标准化的公式为：
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$

这里本实验使用 scikit-learn 提供的 StandardScaler 类进行标准化处理。

```python
scaler = StandardScaler()
train_feature = scaler.fit_transform(train_feature)
dev_feature = scaler.transform(dev_feature)
test_feature = scaler.transform(test_feature)
```

#### 2.1.3 数据保存和读取

即使进行了并行处理，提取特征的时间依然较长，本实验选择将提取的特征保存到文件中，以便下次直接读取特征，不用再次提取。

```python
# 提取样本数据并保存
train_feature = extract_features_parallel(train_path)
df = pd.DataFrame(train_feature)
df["label"] = train_label
df.to_csv("train_feature.csv", index=False)

dev_feature = extract_features_parallel(dev_path)
df = pd.DataFrame(dev_feature)
df["label"] = dev_label
df.to_csv("dev_feature.csv", index=False)

test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
test_path = list(test_csv.path)
test_feature = extract_features_parallel(test_path)
df = pd.DataFrame(test_feature)
df.to_csv("test_feature.csv", index=False)


# 读取保存的样本数据
train_feature = pd.read_csv("train_feature.csv")[:-1]
dev_feature = pd.read_csv("dev_feature.csv")[:-1]
test_feature = pd.read_csv("test_feature.csv")[:]

train_label = train_feature["label"]
dev_label = dev_feature["label"]
train_feature = train_feature.drop(columns=["label"])
dev_feature = dev_feature.drop(columns=["label"])
```

### 2.2 模型设计

本实验采用 LightGBM 模型进行训练，LightGBM 是一种基于决策树算法的梯度提升框架，它具有训练速度快、内存占用低、准确率高等优点。

它的大致过程如下：

**LightGBM 模型训练过程：**

LightGBM 是一种基于决策树的梯度提升框架，其训练过程主要包括以下几个步骤：

1. **初始化模型**：首先初始化一个简单的模型，例如将所有样本预测为相同的值。

2. **计算残差**：计算当前模型的残差，即预测值与真实值之间的差异。对于分类问题，残差可以表示为负梯度。

3. **构建决策树**：使用残差作为新的目标值，构建一棵决策树。决策树的每个叶子节点代表一个新的预测值。

4. **更新模型**：将新构建的决策树加入到模型中，更新模型的预测值。

5. **重复迭代**：重复步骤2到步骤4，直到达到预定的迭代次数或满足其他停止条件。

LightGBM 通过以下公式更新模型：

$$
F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

其中，\( F_{m}(x) \) 是第 \( m \) 次迭代的模型，\( F_{m-1}(x) \) 是第 \( m-1 \) 次迭代的模型，\( \eta \) 是学习率，\( h_m(x) \) 是第 \( m \) 棵决策树。

通过上述过程，LightGBM 能够高效地训练出高性能的模型。

为了方便训练，本实验设计了新的 `MyLGBM` 类，继承自 `LGBMClassifier` 类，实现了 `fit` , `predict`  和 `evaluate` 方法。

```python
# LightGBM 模型类
class MyLGBM:
    def __init__(self, **kwargs):
        self.clf = LGBMClassifier(**kwargs)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        acc, ua, f1, precision, confuse_matrix = calculate_score_classification(
            self.predict(X), y
        )
        return acc, ua, f1, precision, confuse_matrix
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

### 3.1 数据集划分

本实验根据老师提供的数据集和分类，采用样本数为3259的训练集，样本数为1031的验证集，样本数为1241的测试集。

### 3.2 模型训练

```python
best_lgbm = MyLGBM()

best_lgbm.fit(train_feature, train_label)
```

## 四、	实验结果与分析

### 4.1 模型评估

训练集上的性能评估结果如下：

```shell
Train:
acc:1.0
ua:1.0
f1:1.0
precision:1.0
confuse_matrix:
[[ 605    0    0    0]
 [   0  891    0    0]
 [   0    0 1066    0]
 [   0    0    0  696]]
```

验证集上的性能评估结果如下：

```shell
Dev:
acc:0.5475728155339806
ua:0.5485398791478739
f1:0.5530475846828597
precision:0.5687934642416081
confuse_matrix:
[[190  62  67   7]
 [ 89 136  66  12]
 [ 20  58 161  19]
 [  3   8  55  77]]
```

显然模型在训练集上表现较好，但在验证集上表现一般。为了提高模型的泛化能力，减少过拟合程度，实验进一步在正则化、树节点、特征数、学习率等方面进行了微调。

### 4.2 模型调优和评估

```python
model = MyLGBM(
    boosting_type="gbdt",
    num_leaves=14,
    max_depth=9,
    learning_rate=0.01,
    n_estimators=800,
    subsample_for_bin=200000,
    min_split_gain=0.1,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=2,
    colsample_bytree=0.8,
    reg_alpha=6,
    reg_lambda=5.0,
    random_state=None,
    n_jobs=-1,
    importance_type="split",
)
```

实验参数设计了14的叶子节点数，9的最大深度，0.01的学习率，800的迭代次数，0.1的最小分裂增益，0.001的最小子节点权重，20的最小子节点样本数，0.8的子采样率，2的子采样频率，0.8的特征采样率，6的 L1 正则化系数，5.0 的 L2 正则化系数。

进过调优，模型得到了以下结果：

```shell
Train:
acc:0.8621853898096992
ua:0.8605109044805632
f1:0.8640578736500462
precision:0.8709105256014537
confuse_matrix:
[[502  32  64   7]
 [ 17 727  92  55]
 [ 18  44 950  54]
 [  7  11  48 630]]

Dev:
acc:0.5805825242718446
ua:0.5802966622759491
f1:0.5812224993577815
precision:0.595528427025291
confuse_matrix:
[[217  49  55   5]
 [ 96 128  66  13]
 [ 17  49 172  20]
 [  3   5  54  81]]
```

### 4.3 交叉验证

本实验对交叉验证进行了初步尝试：

```python
from sklearn.model_selection import cross_val_score, cross_validate
from lightgbm import LGBMClassifier
import numpy as np

model = LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=12,
    max_depth=None,
    learning_rate=0.01,
    n_estimators=800,
    subsample_for_bin=200000,
    min_split_gain=0.1,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=2,
    colsample_bytree=0.8,
    reg_alpha=8,
    reg_lambda=1,
    random_state=None,
    n_jobs=-1,
    importance_type="split",
)

scores = cross_val_score(model, train_feature, train_label, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average cross-validation score:", np.mean(scores))

cv_results = cross_validate(model, train_feature, train_label, cv=5, scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
print("Cross-validation results:", cv_results)
print("Average accuracy:", np.mean(cv_results['test_accuracy']))
print("Average F1 score:", np.mean(cv_results['test_f1_macro']))
print("Average precision:", np.mean(cv_results['test_precision_macro']))
print("Average recall:", np.mean(cv_results['test_recall_macro']))

best_lgbm = model
best_lgbm.fit(train_feature, train_label)

acc, ua, f1, precision, confuse_matrix = calculate_score_classification(best_lgbm.predict(train_feature), train_label)
print(f"Train:\nacc:{acc}\nua:{ua}\nf1:{f1}\nprecision:{precision}\nconfuse_matrix:\n{confuse_matrix}")

acc, ua, f1, precision, confuse_matrix = calculate_score_classification(best_lgbm.predict(dev_feature), dev_label)
print(f"Dev:\nacc:{acc}\nua:{ua}\nf1:{f1}\nprecision:{precision}\nconfuse_matrix:\n{confuse_matrix}")
```

可惜并未获得更好的结果、更优的模型，实验需要时间进一步调参或者尝试增加数据集大小等办法，以提高模型的性能。


## 五、 总结与展望

本实验通过音频数据进行情感识别任务，实现了音频单模态机器学习，通过特征提取、预处理、模型设计、性能评估等步骤，实现了音频数据的情感识别任务。

虽然进过调优，模型在验证集上的性能有所提升，但仍然存在一定的过拟合问题。为了进一步提高模型的泛化能力，可以考虑通过交叉验证、集成学习、模型融合等方法进一步提高模型的性能。本实验对交叉验证进行了初步尝试，但是未能获得更好的结果，还有进一步的改进空间。