import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer


def create_dataset(dir='data', 
                   train_size=0.8, 
                   val_size=0.1,
                   name_tokenizer='bert-base-uncased'):
    print('\n\tCreate dataset\n')
    def clean_text(text):
        text = text.lower()  # к нижнему регистру
        text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
        text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы
        return text

    def del_ban(text):
        ban = ['@', '://', '/', 'http', 'www', '.com']
        splited = text.split()
        for word in splited:
            for b in ban:
                if b in word:
                    text = text.replace(word, "")
                    break
        return text

    texts = []
    
    with open(dir+'/tweets.txt', 'r', encoding='latin-1') as f:
        prog = tqdm(f.readlines())
        for line in prog:
            clean_1 = del_ban(line)
            clean_2 = clean_text(clean_1)
            texts.append(clean_2)
            
    clean = pd.DataFrame({
        'text': texts
    })
    clean.to_csv(dir+'/clean.csv')

    print('\tShuffle')

    clean_shuffled = clean.sample(frac=1)
    clean_shuffled_list = clean_shuffled['text'].to_list()
    size = len(texts)
    train_end = int(size*train_size)
    val_end = int(size*(train_size+val_size))
    print('\tSplit')
    X_train = clean_shuffled_list[:train_end]
    X_val = clean_shuffled_list[train_end:val_end]
    X_test = clean_shuffled_list[val_end:]
    # X_train, X_val, y_train, y_val = train_test_split(clean_texts, sentiments, train_size=0.8)
    train = pd.DataFrame({
        'text': X_train,
    })    
    val = pd.DataFrame({
        'text': X_val,
    })    
    test = pd.DataFrame({
        'text': X_test,
    })  
    train.to_csv(dir+'/train.csv')
    val.to_csv(dir+'/val.csv')
    test.to_csv(dir+'/test.csv')
    return X_train, X_val, X_test

def tokenize(text, name_tokenizer='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    text_tokenized = tokenizer(text, truncation=True)['input_ids']
    tokenized = pd.DataFrame({
        'text': text_tokenized
    })
    return text_tokenized

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, seq_len=10):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            # if len(token_ids) < seq_len:
            #     continue
            for i in range(1, len(token_ids) - 1):
                context = token_ids[max(0, i - seq_len): i] 
            #     if len(context) < seq_len:
            #         continue
                target = token_ids[i]
                self.samples.append((context, target))
            """l_tokens = len(token_ids)
            context = token_ids[:-1]
            target = token_ids[int(0.75*l_tokens):]
            self.samples.append((context, target))"""

    # возвращаем размер датасета (кол-во текстов)
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {
            'input': torch.tensor(x, dtype=torch.long), 
            'target': torch.tensor(y, dtype=torch.long)
        }
    