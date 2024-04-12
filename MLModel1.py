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
