import torch
from tqdm import tqdm
import evaluate

rouge = evaluate.load("rouge")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(model,
          test_loader,
          tokenizer):
    model = model.to(device)
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            for i in range(len(y_batch)):
                input_tokens = tokenizer.convert_ids_to_tokens(x_batch[i].tolist())
                true_tok = tokenizer.convert_ids_to_tokens([y_batch[i].item()])[0]
                pred_tok = tokenizer.convert_ids_to_tokens([preds[i].item()])[0]
                # print(f"Input: {' '.join(input_tokens)} | True: {true_tok} | Predicted: {pred_tok}")
                predictions.append(input_tokens + ' ' + pred_tok)
                references.append(input_tokens + ' ' + true_tok)
    res_rouge = rouge.compute(predictions=predictions, references=references)
                
    for key, value in res_rouge.items():
        print(f"{key}: {value:.4f}") 
    for i in range(10):
        print("True: {}\nPrediction: {}\n".format(references, predictions))

