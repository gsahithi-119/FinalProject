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


plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_df), color='orange')
plt.bar('Real News', len(real_df), color='green')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News type', size=15)
plt.ylabel('Number of news articles', size=15)
plt.show()
