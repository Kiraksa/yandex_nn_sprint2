from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import evaluate

rouge = evaluate.load("rouge")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_pretrained(
        prompt,
        model_name="distilgpt2",
        tokenizer_name="distilgpt2",
        ):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0  # -1 = CPU; 0 = первый GPU (если есть)
    )
    if isinstance(prompt, str):
        result = generator(prompt, 
                           max_length=20, 
                           do_sample=True, 
                           top_k=50)
        print(result[0]["generated_text"]) 
    elif isinstance(prompt, list):
        for text in prompt:
            result = generator(text, 
                           max_length=20, 
                           do_sample=True, 
                           top_k=50)
            print(result[0]["generated_text"]) 
    else:
        predictions, references = [], []
        with torch.no_grad():
            for batch in prompt:
                x_batch, y_batch = batch['input'].to(device), batch['target'].to(device)
                for i in range(len(y_batch)):
                    input_tokens = tokenizer.convert_ids_to_tokens(x_batch[i].tolist())
                    true_tok = tokenizer.convert_ids_to_tokens([y_batch[i].item()])[0]
                    predict = generator(text, 
                           max_length=20, 
                           do_sample=True, 
                           top_k=50)
                    pred_tok = predict[0]["generated_text"]
                    # print(f"Input: {' '.join(input_tokens)} | True: {true_tok} | Predicted: {pred_tok}")
                    predictions.append(input_tokens + ' ' + pred_tok)
                    references.append(input_tokens + ' ' + true_tok)
        res_rouge = rouge.compute(predictions=predictions, references=references)
        for key, value in res_rouge.items():
            print(f"{key}: {value:.4f}") 
        for i in range(10):
            print("True: {}\nPrediction: {}\n".format(references, predictions))