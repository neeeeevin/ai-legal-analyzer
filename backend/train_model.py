import os
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ==============================
# 1. SETUP
# ==============================
os.makedirs("models/legal_classifier", exist_ok=True)

# ==============================
# 2. LOAD DATASET (BILLSUM)
# ==============================
dataset = load_dataset("billsum", split="train[:1000]")

# ==============================
# 3. BALANCED LABELING (CRITICAL FIX)
# ==============================
summary_lengths = [len(x["summary"].split()) for x in dataset]
threshold = int(np.median(summary_lengths))

print("Balanced threshold:", threshold)

def map_label(example):
    length = len(example["summary"].split())
    return {"label": 1 if length >= threshold else 0}

dataset = dataset.map(map_label)

# ==============================
# 4. STRATIFIED SPLIT (SAFE)
# ==============================
data_list = [dataset[i] for i in range(len(dataset))]
labels = [x["label"] for x in data_list]

train_data, test_data = train_test_split(
    data_list,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

dataset = {
    "train": train_dataset,
    "test": test_dataset
}

# ==============================
# 5. TOKENIZATION
# ==============================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

dataset["train"] = dataset["train"].map(tokenize, batched=True)
dataset["test"] = dataset["test"].map(tokenize, batched=True)

dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ==============================
# 6. MODEL
# ==============================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ==============================
# 7. CLASS WEIGHTS
# ==============================
train_labels = [x["label"] for x in train_data]
class_counts = np.bincount(train_labels)

print("Class distribution:", class_counts)

weights = 1.0 / class_counts
weights = weights / weights.sum()

class_weights = torch.tensor(weights, dtype=torch.float)

print("Class weights:", class_weights)

# ==============================
# 8. CUSTOM TRAINER (FIXED API)
# ==============================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels") if "labels" in inputs else inputs.get("label")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ==============================
# 9. METRICS (BALANCED)
# ==============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    # Slightly lower threshold → better recall
    preds = (probs[:, 1] > 0.4).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=1
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ==============================
# 10. TRAINING CONFIG
# ==============================
training_args = TrainingArguments(
    output_dir="models/legal_model",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    use_cpu=True
)

# ==============================
# 11. TRAIN
# ==============================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()

# ==============================
# 12. SAVE MODEL
# ==============================
trainer.save_model("models/legal_classifier")
tokenizer.save_pretrained("models/legal_classifier")

print("✅ FINAL MODEL TRAINED WITH BALANCED DATA")