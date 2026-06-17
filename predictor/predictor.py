import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel

# Direktori model lokal sentiment_model_v4
MODEL_DIR = os.path.join(os.path.dirname(__file__), "sentiment_model_v4")

_device    = torch.device('cpu')
_tokenizer = None
_model     = None

class IndoBERTEnhanced(nn.Module):
    def __init__(self, model_dir: str, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        # Load config secara offline dari folder lokal
        config = AutoConfig.from_pretrained(model_dir)
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size   # 768

        # Arsitektur head classifier v4 kustom
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits     = self.classifier(cls_output)
        return None, logits

def load_model():
    global _tokenizer, _model
    if _model is None:
        print("🔄 Loading Local IndoBERT v4...")
        # Load tokenizer secara lokal
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        # Inisialisasi arsitektur model kustom
        _model = IndoBERTEnhanced(MODEL_DIR, num_labels=2, dropout=0.2)
        
        # Load bobot latih model lokal
        weights_path = os.path.join(MODEL_DIR, "model_weights.pt")
        _model.load_state_dict(torch.load(weights_path, map_location=_device))
        
        _model.to(_device)
        _model.eval()
        print("✅ Local Model loaded!")
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
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred      = torch.argmax(logits, dim=1).item()

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
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs     = torch.softmax(logits, dim=1).cpu().numpy()
            preds     = torch.argmax(logits, dim=1).cpu().numpy()

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