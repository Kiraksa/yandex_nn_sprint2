from src.data_utils import create_dataset, LSTMDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from src.lstm_model import MyLSTM
from src.lstm_train import train as train_lstm
from src.eval_lstm import eval as eval_lstm

train, val, test = create_dataset()

name_tokenizer='bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)

train_dataset = LSTMDataset(train, tokenizer, seq_len=20)
val_dataset = LSTMDataset(val, tokenizer, seq_len=20)

def collate_fn(batch):
    texts = [item['input'] for item in batch]
    labels =  torch.tensor([item['target'] for item in batch], dtype=torch.long)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0) 


    return {
        'input': padded_texts, 
        'target': labels
    }

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=300, shuffle=False, collate_fn=collate_fn)

vocab_size = tokenizer.vocab_size  
hidden_dim = 128
model = MyLSTM(vocab_size, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch = 6
model = model.to(device)
train_lstm(
    model,
    train_loader,
    optimizer,
    tokenizer,
    val_loader,
    criterion,
    n_epoch
)