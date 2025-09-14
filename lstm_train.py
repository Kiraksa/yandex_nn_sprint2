import torch
from tqdm import tqdm
import evaluate
import random

rouge = evaluate.load("rouge")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(model,
          test_loader,
          tokenizer,
          criterion=torch.nn.CrossEntropyLoss()):
    model = model.to(device)
    model.eval()
    sum_loss = 0
    predictions, references = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            preds = torch.argmax(x_output, dim=1)
            for i in range(len(y_batch)):
                input_tokens = tokenizer.convert_ids_to_tokens(x_batch[i].tolist())
                true_tok = tokenizer.convert_ids_to_tokens([y_batch[i].item()])[0]
                pred_tok = tokenizer.convert_ids_to_tokens([preds[i].item()])[0]
                predictions.append(input_tokens + ' ' + pred_tok)
                references.append(input_tokens + ' ' + true_tok)
            res_rouge = rouge.compute(predictions=predictions, references=references)
            sum_loss += loss.item()
    return {
        'loss': sum_loss / len(test_loader),
        'rouge': res_rouge,
        'predictions': predictions,
        'references': references
    } 

def train(model,
          train_loader,
          optimizer,
          tokenizer,
          val_loader=None,
          criterion=torch.nn.CrossEntropyLoss(),
          n_epochs=20):
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        train_loss /= len(train_loader)
        if val_loader is not None:
            val_res = eval(model, val_loader, tokenizer, criterion)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_res['loss']:.3f}")
            for key, value in val_res['rouge'].items():
                print(f"{key}: {value:.4f}") 
        else:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f}")
    for i in range(5):
        print("True: {}\nPrediction: {}\n".format(val_res['references'][i], val_res['predictions'][i]))