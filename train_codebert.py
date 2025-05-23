from datasets import load_dataset
from transformers import (RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

data_files = {
    "train": "splits/train.jsonl",
    "validation": "splits/val.jsonl"
}

ds = load_dataset("json", data_files=data_files)

ds = ds.map(tokenize, batched=True)

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

args = TrainingArguments(
    output_dir="codebert-ft",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

print("==== TRAINING METRICS ====")
print(trainer.evaluate(ds["train"]))

print("==== VALIDATION METRICS ====")
print(trainer.evaluate(ds["validation"]))

model.save_pretrained("codebert-ft/best")
tokenizer.save_pretrained("codebert-ft/best")