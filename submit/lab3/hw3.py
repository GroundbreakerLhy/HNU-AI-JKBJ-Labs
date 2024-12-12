import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import LlamaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


# 设置随机种子
def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


set_seeds(42)

# 加载数据
train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")
test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")

train_txt = list(train_csv.text)
train_label = list(train_csv.label)
dev_txt = list(dev_csv.text)
dev_label = list(dev_csv.label)
test_txt = list(test_csv.text)

# 加载本地的 LLaMA 分词器
# 请将 "./llama" 替换为您本地 LLaMA 模型的路径
tokenizer = AutoTokenizer.from_pretrained("llama-3.2-1B")

# 设置 pad_token_id，否则会出现错误
tokenizer.pad_token_id = tokenizer.eos_token_id

# 定义标签数量
num_labels = 4

# 加载 LLaMA 模型并添加分类头
model = LlamaForSequenceClassification.from_pretrained(
    "./llama-3.2-1B",
    num_labels=num_labels,
    ignore_mismatched_sizes=True,  # 忽略形状不匹配的警告
)
model.config.pad_token_id = tokenizer.pad_token_id
model.pad_token_id = tokenizer.pad_token_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print(f'Using {torch.cuda.device_count()} GPUs.')
#     model = torch.nn.DataParallel(model)
model.to(device)


# 文本编码函数
def text_tokenize(text_list):
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    return encoded_text


# 编码文本
train_encoded = text_tokenize(train_txt)
dev_encoded = text_tokenize(dev_txt)
test_encoded = text_tokenize(test_txt)

# 创建数据集
train_dataset = TensorDataset(
    train_encoded["input_ids"],
    train_encoded["attention_mask"],
    torch.tensor(train_label),
)
dev_dataset = TensorDataset(
    dev_encoded["input_ids"], dev_encoded["attention_mask"], torch.tensor(dev_label)
)
test_dataset = TensorDataset(test_encoded["input_ids"], test_encoded["attention_mask"])

# 创建数据加载器
batch_size = 32  # 根据显存大小调整
dataloader_train = DataLoader(
    train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
)
dataloader_dev = DataLoader(
    dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size
)
dataloader_test = DataLoader(
    test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
)


# 定义模型训练类
class MyDLmodel:
    def __init__(self, model, device, patience=3):
        self.device = device
        self.patience = patience
        self.best_f1_score = 0
        self.early_stop_counter = 0

        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.05)

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
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(dataloader_train)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

            val_loss, val_f1 = self.evaluate(dataloader_dev)

            # 早停机制
            if val_f1 > self.best_f1_score:
                self.best_f1_score = val_f1
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print("Validation F1 Score improved, saving model.\n")
            else:
                self.early_stop_counter += 1
                print(
                    f"No improvement in validation F1 Score for {self.early_stop_counter} epoch(s).\n"
                )
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at F1 Score: {self.best_f1_score:.4f}")
                    break

    def evaluate(self, dataloader_val):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader_val:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_loss / len(dataloader_val)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}"
        )

        return avg_val_loss, f1

    def predict(self, dataloader_test):
        self.model.eval()
        self.model.load_state_dict(torch.load("best_model.pt"))

        all_preds = []
        with torch.no_grad():
            for batch in dataloader_test:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask = batch

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return all_preds


# 定义训练参数
epochs = 15  # 根据需要调整
mymodel = MyDLmodel(model, device)

# 开始训练
mymodel.train(dataloader_train, dataloader_dev, epochs)

# # 在测试集上进行预测
# test_preds = mymodel.predict(dataloader_test)


# # 将预测结果写入文件
# def write_result(test_preds):
#     if len(test_preds) != len(test_csv):
#         print("错误！请检查 test_preds 长度是否正确！")
#         return -1
#     test_csv["label"] = test_preds
#     test_csv.to_csv("result.csv", sep="#", index=False)
#     print("测试集预测结果已成功写入到文件中！")


# write_result(test_preds)
