import pandas as pd
import json
from tqdm import tqdm

input_files = ["EmotionLines/Friends/friends_train.json", "EmotionLines/Friends/friends_dev.json", "EmotionLines/Friends/friends_test.json"]

for file in input_files:
    with open(file, "r") as f:
        data = json.load(f)
        train_data = {"text": [], "label": []}
        for d in data:
            for single in tqdm(d):
                train_data["text"].append(single["utterance"])
                if single["emotion"] in ["anger"]:
                    train_data["label"].append(0)
                elif single["emotion"] in ["joy", "suprise", "non-neutral"]:
                    train_data["label"].append(1)
                elif single["emotion"] in ["neutral"]:
                    train_data["label"].append(2)
                else:
                    train_data["label"].append(3)
df = pd.DataFrame(train_data)
df.to_csv("CSVfile/friends.csv", sep="#", index=False)
# emotions = set()
# for file in input_files:
#     with open(file, "r") as f:
#         data = json.load(f)
#         for d in data:
#             for single in d:
#                 emotions.add(single["emotion"])
# print(emotions)