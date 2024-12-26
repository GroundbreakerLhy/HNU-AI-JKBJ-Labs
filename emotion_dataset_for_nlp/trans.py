import pandas as pd
from tqdm import tqdm

files = ["emotion_dataset_for_nlp/val.txt", "emotion_dataset_for_nlp/test.txt", "emotion_dataset_for_nlp/train.txt"]
emotions = set()
data_to_write = {}
for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            text = line.strip().split(";")[0]
            label = line.strip().split(";")[1]
            if label in ["anger"]:
                label = 0
            elif label in ["joy", "love", "surprise"]:
                label = 1
            elif label in ["sadness", "fear"]:
                label = 3
            data_to_write[text] = label
df = pd.DataFrame(data_to_write.items(), columns=["text", "label"])
df.to_csv("CSVfile/emotion_data.csv", sep="#", index=False)
#             emotion = line.strip().split(";")[1]
#             emotions.add(emotion)
# print(emotions)