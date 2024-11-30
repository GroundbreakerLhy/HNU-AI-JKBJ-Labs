import opensmile
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from concurrent.futures import ThreadPoolExecutor, as_completed


# 提取特征函数
def extract_audio_feature(file):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    y = smile.process_file(file)
    y = y.to_numpy().reshape(-1)
    return y


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


# 计算性能指标函数
def calculate_score_classification(preds, labels, average_f1="macro"):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    ua = recall_score(labels, preds, average="macro", zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix


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


# # 提取训练样本标签
# train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
# dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")

# train_path = list(train_csv.path)
# train_label = list(train_csv.label)
# dev_path = list(dev_csv.path)
# dev_label = list(dev_csv.label)

# # 提取训练样本特征
# train_feature = extract_features_parallel(train_path)
# df = pd.DataFrame(train_feature)
# df["label"] = train_label
# df.to_csv("./CSVfile/train_feature.csv", index=False)

# dev_feature = extract_features_parallel(dev_path)
# df = pd.DataFrame(dev_feature)
# df["label"] = dev_label
# df.to_csv("./CSVfile/dev_feature.csv", index=False)

# test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
# test_path = list(test_csv.path)
# test_feature = extract_features_parallel(test_path)
# df = pd.DataFrame(test_feature)
# df.to_csv("./CSVfile/test_feature.csv", index=False)

# 加载特征和标签
train_feature = pd.read_csv("./CSVfile/train_feature.csv")[:-1]
dev_feature = pd.read_csv("./CSVfile/dev_feature.csv")[:-1]
test_feature = pd.read_csv("./CSVfile/test_feature.csv")[:]

train_label = train_feature["label"]
dev_label = dev_feature["label"]
train_feature = train_feature.drop(columns=["label"])
dev_feature = dev_feature.drop(columns=["label"])

# 特征标准化
scaler = StandardScaler()
train_feature = scaler.fit_transform(train_feature)
dev_feature = scaler.transform(dev_feature)
test_feature = scaler.transform(test_feature)
# 训练模型
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

best_lgbm = model
best_lgbm.fit(train_feature, train_label)

# 评估模型在训练集上的表现
acc, ua, f1, precision, confuse_matrix = calculate_score_classification(best_lgbm.predict(train_feature), train_label)
print(f"Train:\nacc:{acc}\nua:{ua}\nf1:{f1}\nprecision:{precision}\nconfuse_matrix:\n{confuse_matrix}")

# 评估模型在验证集上的表现
acc, ua, f1, precision, confuse_matrix = calculate_score_classification(best_lgbm.predict(dev_feature), dev_label)
print(f"Dev:\nacc:{acc}\nua:{ua}\nf1:{f1}\nprecision:{precision}\nconfuse_matrix:\n{confuse_matrix}")

# 结果写入函数
def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误!请检查test_preds长度是否为1241!!!")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./submit/lab2/hw2_result.csv", sep="#")
    print("测试集预测结果已成功写入到文件中!")

# 预测测试集并写入结果
test_preds = best_lgbm.predict(test_feature)
write_result(test_preds)
