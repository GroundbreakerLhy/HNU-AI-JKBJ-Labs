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
import torch.nn.functional as F
import torch.nn as nn
import nltk
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")
nltk.download("wordnet")


def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def add_prompt(
    text_list,
    prompt="""I am working on emotion classification, please make an emotion classification based on the following text. 
            The classification label 0 means ANGRY, 1 is HAPPY or EXCITED, 2 is NEUTRAL, 3 is SAD: \n""",
):
    return [prompt + text for text in text_list]


def text_tokenize(text_list, tokenizer):
    encoded_text = tokenizer.batch_encode_plus(
        text_list,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    return encoded_text


def write_result(test_preds):
    if len(test_preds) != 1241:
        print("错误!请检查test_preds长度是否为1241!!!")
        return -1
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")
    test_csv["label"] = test_preds
    test_csv.to_csv("./submit/lab2/hw2_result.csv", sep="#")
    print("测试集预测结果已成功写入到文件中!")


def synonym_replacement(text_list, n):
    random.seed(42)
    augmented_text = []
    for sentence in text_list:
        words = sentence.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = wordnet.synsets(random_word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                new_words = [
                    synonym if word == random_word else word for word in new_words
                ]
                num_replaced += 1
            if num_replaced >= n:
                break
        augmented_sentence = " ".join(new_words)
        augmented_text.append(augmented_sentence)
    return augmented_text


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


class MyDLmodel:
    def __init__(self, model, device, patience=3):
        self.device = device
        self.patience = patience
        self.best_f1_score = 0
        self.early_stop_counter = 0

        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=8e-6,
            eps=1e-8,
            weight_decay=0.1,
            betas=(0.9, 0.999),
        )
        self.smoothing = 0.1
        self.criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)

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
                torch.save(self.model.state_dict(), "best_model.pt")
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
                logits = outputs.logits
                loss = self.criterion(logits, labels)

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

        return f1

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


if __name__ == "__main__":
    set_seeds(42)

    train_csv = pd.read_csv("./CSVfile/train.csv", sep="#")
    dev_csv = pd.read_csv("./CSVfile/dev.csv", sep="#")
    test_csv = pd.read_csv("./CSVfile/test.csv", sep="#")

    train_txt = list(train_csv.text)
    train_label = list(train_csv.label)
    dev_txt = list(dev_csv.text)
    dev_label = list(dev_csv.label)
    test_txt = list(test_csv.text)

    train_txt_augmented = synonym_replacement(train_txt, 2)
    train_txt = train_txt + train_txt_augmented
    train_label = train_label + train_label

    tokenizer = AutoTokenizer.from_pretrained("llama-3.2-1B")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    num_labels = 4
    model = LlamaForSequenceClassification.from_pretrained(
        "./llama-3.2-1B",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_txt = add_prompt(train_txt)
    dev_txt = add_prompt(dev_txt)
    test_txt = add_prompt(test_txt)

    train_encoded = text_tokenize(train_txt, tokenizer)
    dev_encoded = text_tokenize(dev_txt, tokenizer)
    test_encoded = text_tokenize(test_txt, tokenizer)

    train_dataset = TensorDataset(
        train_encoded["input_ids"],
        train_encoded["attention_mask"],
        torch.tensor(train_label),
    )
    dev_dataset = TensorDataset(
        dev_encoded["input_ids"], dev_encoded["attention_mask"], torch.tensor(dev_label)
    )
    test_dataset = TensorDataset(
        test_encoded["input_ids"], test_encoded["attention_mask"]
    )

    batch_size = 48
    dataloader_train = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    dataloader_dev = DataLoader(
        dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size
    )
    dataloader_test = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
    )

    epochs = 15
    mymodel = MyDLmodel(model, device)
    mymodel.train(dataloader_train, dataloader_dev, epochs)

    # test_preds = mymodel.predict(dataloader_test)
    # write_result(test_preds)
