import os
import threading
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from preprocessing.sentiment_cleaning import clean_text as clean_for_sentiment

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "Pipeline_IndoBERT_Training_dan_Validasi_Manual_AppStore_v2",
    "sentiment_model_appstore_v1",
)

_device = torch.device("cpu")
_tokenizer = None
_model = None
_model_lock = threading.Lock()


class IndoBERTEnhanced(nn.Module):
    def __init__(self, model_dir: str, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size

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
        logits = self.classifier(cls_output)
        return None, logits


def load_model():
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _device

    with _model_lock:
        if _model is None or _tokenizer is None:
            print("🔄 Loading Local IndoBERT (App Store)...")

            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR,
                local_files_only=True
            )

            _model = IndoBERTEnhanced(MODEL_DIR, num_labels=2, dropout=0.2)

            weights_path = os.path.join(MODEL_DIR, "model.pt")
            state_dict = torch.load(weights_path, map_location=_device)
            _model.load_state_dict(state_dict)

            _model.to(_device)
            _model.eval()
            print("✅ Local Model loaded!")

    return _model, _tokenizer, _device


print("🚀 Preloading model...")
load_model()
print("✅ Model siap!")


def predict_sentiment(text: str) -> dict:
    if not text or not str(text).strip():
        return {
            "label": "negatif",
            "confidence": 0.0,
            "scores": {
                "negatif": 0.0,
                "positif": 0.0,
            }
        }

    model, tokenizer, device = load_model()

    encoding = tokenizer(
        clean_for_sentiment(text),
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.inference_mode():
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(logits, dim=1).item())

    label_map = {0: "negatif", 1: "positif"}

    return {
        "label": label_map[pred],
        "confidence": round(float(probs[pred]) * 100, 2),
        "scores": {
            "negatif": round(float(probs[0]) * 100, 2),
            "positif": round(float(probs[1]) * 100, 2),
        }
    }


def predict_batch(texts: list, batch_size: int = 32) -> list:
    if not texts:
        return []

    model, tokenizer, device = load_model()
    label_map = {0: "negatif", 1: "positif"}
    results = []

    total_batches = -(-len(texts) // batch_size)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"   🔄 Batch {i // batch_size + 1} / {total_batches} ({len(batch_texts)} teks)")

        clean_batch = [clean_for_sentiment(t) for t in batch_texts]

        encoding = tokenizer(
            clean_batch,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.inference_mode():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        for prob, pred in zip(probs, preds):
            pred = int(pred)
            results.append({
                "label": label_map[pred],
                "confidence": round(float(prob[pred]) * 100, 2),
                "scores": {
                    "negatif": round(float(prob[0]) * 100, 2),
                    "positif": round(float(prob[1]) * 100, 2),
                }
            })

    return results