import pandas as pd 
import matplotlib.pyplot as plt 
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from datasets import load_metric
import numpy as np 


all_df = pd.read_csv('RealFakeShort.csv').dropna()
fake_df, real_df = all_df[all_df['label'] == 0], all_df[all_df['label'] == 1]

# Create a plot showing data distribution
plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_df), color='red')
plt.bar('Real News', len(real_df), color='green')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News type', size=15)
plt.ylabel('Number of news articles', size=15)
plt.show()

# Divide training data into training, testing, evaluation sets
X_train, X_test, y_train, y_test = train_test_split(all_df['title'], all_df['label'], test_size=0.20, random_state=15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=15)

# Get DistilBERT model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create class News
class News(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    # convert to pytorch tensors
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item 
    def __len__(self):
        return len(self.labels)

sample_texts = [str(text) for text in X_train[:5]]

# Error handling
try:
    sample_encodings = tokenizer(sample_texts, truncation=True, padding=True)
    print("Sample tokenization successful:", sample_encodings.keys())
except Exception as e:
    print("Detailed Error during tokenization:", e)
