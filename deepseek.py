import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from collections import Counter
import random
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, precision_score, recall_score
import re

# ========== Config ==========
MODEL_NAME = os.path.expanduser("~/DeepSeek-R1-Distill-Qwen-32B")  # Change as needed (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
OUTPUT_DIR = os.path.expanduser("~/DeepSeek-R1-Distill-Qwen-32B_lora_adapterss")
TRAIN_FILE = os.path.expanduser("~/downsampled_cookies_training_data.jsonl")
VAL_FILE = os.path.expanduser("~/downsampled_cookies_validation_data.jsonl")

train_raw = load_dataset("json", data_files=os.path.expanduser("~/downsampled_cookies_training_data.jsonl"))["train"]
val_raw = load_dataset("json", data_files=os.path.expanduser("~/downsampled_cookies_validation_data.jsonl"))["train"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = DatasetDict({'train':train_raw, 'validation':val_raw})
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 4096

def to_chat_template(example):
    try:
        output_dict = json.loads(example["output"])
        value = output_dict.get("Cookies_and_tracking_elements", "").strip().upper()
    except json.JSONDecodeError:
        value = example["output"].strip().upper()

    # Compose the long policy text from example['prompt'] (assuming that's your policy+question)
    long_text = example["prompt"]
    cleaned_text = re.sub(r'\s+', ' ', long_text).strip()
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                'Read the privacy policy and answer ONLY "YES" or "NO".\n'
                "Does the following policy mentions that it shares cookies and tracking elements with third parties?\n"
                f"{cleaned_text}"
            )
        },
        {
            "role": "assistant",
            "content": value
        }
    ]

    label = 1 if "YES" in value else 0
    return {"text": chat, "label": label}

# Map dataset
dataset = dataset.map(to_chat_template)
train_labels = [x['label'] for x in dataset["train"]]
val_labels = [x['label'] for x in dataset["validation"]]

train_counter = Counter(train_labels)
val_counter = Counter(val_labels)

# Weighted sampling
labels = dataset['train']['label']
class_counts = torch.bincount(torch.tensor(labels))
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)

# Render template string
def apply_chat_temp(example):
    rendered = tokenizer.apply_chat_template(example['text'], tokenize=False)
    return {"text": rendered}

dataset = dataset.map(apply_chat_temp)

print(dataset['train'][0]['text'])
template_ids = tokenizer.encode(dataset['train'][0]['text'], add_special_tokens=False)
print(template_ids)
response_template = "<｜Assistant｜>"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
print(response_template_ids)
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer
)

def tokenize_fn(example):
    return tokenizer(example['text'], max_length=4096, add_special_tokens=False)


tokenized_dataset = dataset.map(tokenize_fn,batched=True,remove_columns=['prompt', 'text', 'output', 'label'])

# ========== QLoRA Config ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=4, # As my train dataset is really small, so i opted for a small value of r.
    lora_alpha=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    use_cache=False,
    quantization_config=bnb_config
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count())
    model.is_parallelizable = True
    model.model_parallel = True

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=6,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    fp16=False,
    bf16=True,
    save_total_limit=3,
    logging_steps=25,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    remove_unused_columns=False,
    logging_first_step=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
)

# ========== Metrics ==========
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    if isinstance(preds, tuple) or isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, tuple) or isinstance(labels, list):
        labels = np.array(labels)
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    preds = preds.astype(np.int32)
    vocab_size = len(tokenizer)
    print(f"[DEBUG] preds min: {preds.min()}, max: {preds.max()}, labels min: {labels.min()}, max: {labels.max()}, vocab_size: {vocab_size}")
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    try:
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Failed to decode preds: {e}")
        print(f"[ERROR] Example preds: {preds[:5]}")
        pred_texts = [""] * len(preds)
    try:
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Failed to decode labels: {e}")
        print(f"[ERROR] Example labels: {labels[:5]}")
        label_texts = [""] * len(labels)
    print("Sample pred_texts:", pred_texts[:5])
    print("Sample label_texts:", label_texts[:5])
    def parse_answer(text):
        match = re.search(r'<｜Assistant｜>\s*(.*?)(</s>|$)', text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip().upper()
            if "YES" in answer:
                return 1
            elif "NO" in answer:
                return 0
        text = text.upper()
        if "YES" in text:
            return 1
        elif "NO" in text:
            return 0
        else:
            return -1
    y_pred = [parse_answer(t) for t in pred_texts]
    y_true = [parse_answer(t) for t in label_texts]
    filtered = [(p, t) for p, t in zip(y_pred, y_true) if p != -1 and t != -1]
    if not filtered:
        return {"accuracy": 0, "f1_macro": 0, "f1_micro": 0, "precision_macro": 0, "recall_macro": 0}
    y_pred, y_true = zip(*filtered)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

# ========== SFTTrainer ==========

class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,  # our custom sampler
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

trainer = WeightedSamplerTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========== Train ==========
print("Starting SFT training with QLoRA and class weights...")
trainer.train()

# ========== Save Model ==========
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapters saved to {OUTPUT_DIR}")

train_metrics = trainer.evaluate(trainer.train_dataset)
print("Training set metrics:", train_metrics)
