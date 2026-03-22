from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

DEVICE = torch.device("cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("models/legal_classifier")
model = DistilBertForSequenceClassification.from_pretrained("models/legal_classifier")

model.to(DEVICE)
model.eval()

id2label = model.config.id2label if hasattr(model.config, "id2label") else None


def classify_batch(texts, batch_size=16):

    if not texts:
        return []

    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        valid_texts = [t for t in batch_texts if isinstance(t, str) and len(t.strip()) > 5]

        if not valid_texts:
            continue

        inputs = tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)

        for j, text in enumerate(valid_texts):
            prob_vector = probs[j].cpu().numpy()
            pred_idx = int(prob_vector.argmax())
            confidence = float(prob_vector[pred_idx])

            if id2label:
                label = id2label.get(pred_idx, "low").lower()
            else:
                label = "low"

            if "high" in label:
                final_label = "high"
            elif "medium" in label:
                final_label = "medium"
            else:
                final_label = "low"

            all_results.append({
                "text": text,
                "label": final_label,
                "confidence": round(confidence, 4)
            })

    return all_results