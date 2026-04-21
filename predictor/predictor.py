import torch
import os
from transformers import BertTokenizer, AutoModelForSequenceClassification

HF_REPO_ID = "soursupz/indobert-shopee-sentiment"
HF_TOKEN   = os.environ.get("HF_TOKEN")

_device    = torch.device('cpu')
_tokenizer = None
_model     = None

def load_model():
    global _tokenizer, _model
    if _model is None:
        print("🔄 Loading IndoBERT dari HuggingFace...")
        _tokenizer = BertTokenizer.from_pretrained(
            HF_REPO_ID,
            token=HF_TOKEN
        )
        _model = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO_ID,
            token=HF_TOKEN,
            low_cpu_mem_usage=True
        )
        _model.to(_device)
        _model.eval()
        print("✅ Model loaded!")
    return _model, _tokenizer, _device

print("🚀 Preloading model...")
load_model()
print("✅ Model siap!")


def predict_sentiment(text: str) -> dict:
    model, tokenizer, device = load_model()

    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred    = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: 'negatif', 1: 'positif'}

    return {
        'label':      label_map[pred],
        'confidence': round(float(probs[pred]) * 100, 2),
        'scores': {
            'negatif': round(float(probs[0]) * 100, 2),
            'positif': round(float(probs[1]) * 100, 2),
        }
    }


def predict_batch(texts: list, batch_size: int = 32) -> list:
    """
    Proses semua teks sekaligus dalam batch — jauh lebih cepat dari 1-per-1.
    batch_size=32 aman untuk CPU Railway (tidak OOM).
    """
    if not texts:
        return []

    model, tokenizer, device = load_model()
    label_map = {0: 'negatif', 1: 'positif'}
    results   = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        print(f"   🔄 Batch {i//batch_size + 1} / {-(-len(texts)//batch_size)} ({len(batch_texts)} teks)")

        encoding = tokenizer(
            batch_texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        for prob, pred in zip(probs, preds):
            results.append({
                'label':      label_map[int(pred)],
                'confidence': round(float(prob[pred]) * 100, 2),
                'scores': {
                    'negatif': round(float(prob[0]) * 100, 2),
                    'positif': round(float(prob[1]) * 100, 2),
                }
            })

    return results