import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from transformers import TrainerCallback
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from torch.nn import CrossEntropyLoss

from collections import Counter
import random
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from sklearn.metrics import f1_score, precision_score, recall_score
import re

# ========== Config ==========
MODEL_NAME = os.path.expanduser("~/Mixtral-8x7B-Instruct-v0.1")  # Change as needed (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
OUTPUT_DIR = os.path.expanduser("~/lora-Mixtral-8x7B-Instruct-v0.1_2e5")
TRAIN_FILE = os.path.expanduser("~//downsampled_processed_training_data.jsonl")
VAL_FILE = os.path.expanduser("~/downsampled_processed_validation_data.jsonl")

train_raw = load_dataset("json", data_files=TRAIN_FILE)["train"]
val_raw = load_dataset("json", data_files=VAL_FILE)["train"]
os.makedirs(OUTPUT_DIR, exist_ok=True)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = DatasetDict({'train':train_raw, 'validation':val_raw})
# ========== Tokenizer ==========

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 1024

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

# --- CRUCIAL: Shuffle the training data before creating the sampler ---
import random

random.seed(42)
dataset["train"] = dataset["train"].select(
    random.sample(range(len(dataset["train"])), len(dataset["train"]))
)

train_labels = [x['label'] for x in dataset["train"]]
val_labels = [x['label'] for x in dataset["validation"]]

train_counter = Counter(train_labels)
val_counter = Counter(val_labels)

# Weighted sampling
labels = dataset['train']['label']
labels_tensor = torch.tensor(labels)

class_counts = torch.bincount(labels_tensor)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[labels_tensor]
print(sample_weights)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


# Render template string
def apply_chat_temp(example):
    rendered = tokenizer.apply_chat_template(example['text'], tokenize=False)
    return {"text": rendered}

dataset = dataset.map(apply_chat_temp)
response_token = "[/INST]"

response_template_ids_for_inst= tokenizer.encode(response_token, add_special_tokens=False)

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids_for_inst,
    tokenizer=tokenizer
)

def tokenize_fn(example):
    return tokenizer(example['text'], max_length=4096, add_special_tokens=False)

tokenized_dataset = dataset.map(tokenize_fn,batched=True,remove_columns=['prompt', 'output', 'text','label'])
# ========== QLoRA Config ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=4, # As my train dataset is really small, so i opted for a small value of r.
    lora_alpha=8,
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
    # model.is_parallelizable = True
    # model.model_parallel = True

# ========== Training Arguments ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=12,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=8e-6,
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
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    logging_first_step=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    optim="adamw_torch",
    eval_accumulation_steps=8
)

# ========== Custom Trainer for Multi-GPU fix and Weighted Sampling ==========
class CustomMultiGPUTrainer(Trainer):
    def __init__(self, *args, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation to handle multi-GPU device mismatches by ensuring
        labels are on the same device as the logits before calculating the loss.
        """
        # We don't pass labels to the model, so it returns logits instead of loss.
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        
        logits = outputs.get("logits")
        
        # The data collator should have created `labels` from `input_ids`.
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels not found in inputs. Check your data collator.")

        # Shift so that tokens < n predict n.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Move labels to the same device as the logits. This is the fix.
        shift_labels = shift_labels.to(shift_logits.device)
        
        # Flatten the tokens and compute the loss, ignoring the padding tokens.
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ========== Memory-clearing callback ==========
class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
# ========== SFTTrainer ==========

trainer = CustomMultiGPUTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    sampler=sampler,
    callbacks=[ClearCacheCallback]
)

# ========== Train ==========
print("Starting SFT training with QLoRA, class weights, and completion-only LM...")
trainer.train()

# ========== Save Model ==========
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapters saved to {OUTPUT_DIR}")

train_metrics = trainer.evaluate(trainer.train_dataset)
print("Training set metrics:", train_metrics)
