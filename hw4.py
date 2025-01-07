import opensmile
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def text_tokenize(text_list):
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased", do_lower_case=True)
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
    )
    return encoded_text


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


def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误！请检查test_preds长度是否为1241！")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./result.csv", sep="#")
    print("测试集预测结果已成功写入到文件中！")


def scale(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


class MultiModalDataset(Dataset):
    def __init__(self, text_features, audio_features, labels=None):
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.text_features["input_ids"])

    def __getitem__(self, idx):
        if self.labels is not None:
            return {
                "text_ids": self.text_features["input_ids"][idx],
                "text_mask": self.text_features["attention_mask"][idx],
                "audio": torch.tensor(self.audio_features[idx], dtype=torch.float32),
                "labels": torch.tensor(self.labels[idx]),
            }
        else:
            return {
                "text_ids": self.text_features["input_ids"][idx],
                "text_mask": self.text_features["attention_mask"][idx],
                "audio": torch.tensor(self.audio_features[idx], dtype=torch.float32),
            }


class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.25):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class HybridFusion(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1024, num_layers=5):
        super().__init__()
        # LSTM 处理序列依赖
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Transformer 处理模态间交互
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        trans_out = self.transformer(lstm_out)

        output = self.norm(lstm_out + trans_out)
        return output


class MultiModalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.bert = BertModel.from_pretrained("./bert-base-uncased")
        self.audio_fc = nn.Linear(88, 256)
        self.text_proj = nn.Linear(768, 256)

        self.fusion = HybridFusion(256, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, text_ids, text_mask, audio):
        text_feat = self.text_proj(self.bert(text_ids, attention_mask=text_mask)[1])
        audio_feat = self.audio_fc(audio)

        combined = torch.stack([text_feat, audio_feat], dim=1)

        fused = self.fusion(combined)
        fused = torch.mean(fused, dim=1)

        return self.classifier(fused)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EmotionClassification:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiModalModel().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            eps=1e-8,
            weight_decay=0.15,
        )
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.15)
        self.patience = 3
        self.best_f1_score = 0
        self.early_stop_counter = 0

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(text_ids, text_mask, audio)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_acc, val_f1, val_cm = self.evaluate(val_loader)
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Confusion Matrix:\n{val_cm}")
            if val_f1 > self.best_f1_score:
                self.best_f1_score = val_f1
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print("F1 Score improved, saving model.\n")
            # else:
            #     self.early_stop_counter += 1
            #     print(f"No improvement for {self.early_stop_counter} epoch(s).\n")
            #     if self.early_stop_counter >= self.patience:
            #         print(f"Early stopping at F1: {self.best_f1_score:.4f}")
            #         break
            print("-" * 50)
        print(f"Training finished. Dev F1: {self.best_f1_score:.4f}")

    def evaluate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(text_ids, text_mask, audio)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return (
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"),
            confusion_matrix(all_labels, all_preds),
        )

    def predict(self, test_loader):
        self.model.eval()
        self.model.load_state_dict(torch.load("best_model.pt"))
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                audio = batch["audio"].to(self.device)

                outputs = self.model(text_ids, text_mask, audio)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions


if __name__ == "__main__":
    set_seeds(42)

    train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
    dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")

    train_text = text_tokenize(train_csv.text.tolist())
    train_audio = pd.read_csv("CSVfile/train_feature.csv").iloc[:, :-1].to_numpy()
    train_audio = scale(train_audio)
    train_label = train_csv.label.tolist()

    dev_text = text_tokenize(dev_csv.text.tolist())
    dev_audio = pd.read_csv("CSVfile/dev_feature.csv").iloc[:, :-1].to_numpy()
    dev_audio = scale(dev_audio)
    dev_label = dev_csv.label.tolist()

    test_audio = pd.read_csv("CSVfile/test_feature.csv").iloc[:, :-1].to_numpy()

    test_text = text_tokenize(test_csv.text.tolist())
    test_audio = pd.read_csv("CSVfile/test_feature.csv").to_numpy()
    test_audio = scale(test_audio)

    train_dataset = MultiModalDataset(train_text, train_audio, train_label)
    dev_dataset = MultiModalDataset(dev_text, dev_audio, dev_label)
    test_dataset = MultiModalDataset(test_text, test_audio)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = EmotionClassification()
    model.train(train_loader, dev_loader, epochs=25)

    model.model.load_state_dict(torch.load("best_model.pt"))
    test_preds = model.predict(test_loader)
    write_result(test_preds)
