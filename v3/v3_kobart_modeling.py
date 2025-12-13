# ===============================================================
# v3_kobart_modeling_fix.py
# KoBART-base-v2 + LoRA + Optuna (seed별 trial 수행)
# - 버전3 목적 유지
# - 기존 문제만 정확히 해결한 안정 버전
# ===============================================================

import os
import gc
import yaml
import torch
import optuna
import random
import numpy as np
import pandas as pd
import wandb

from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer
from peft import LoraConfig, get_peft_model, TaskType


# ===============================================================
# 0. Load Config
# ===============================================================
def load_config(path: str = "v3_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

DATA_DIR   = cfg["general"]["data_dir"]
TRAIN_FILE = os.path.join(DATA_DIR, cfg["general"]["train_file"])
DEV_FILE   = os.path.join(DATA_DIR, cfg["general"]["dev_file"])
MODEL_NAME = cfg["general"]["model_name"]
OUTPUT_DIR = cfg["general"]["output_dir"]
SEED_LIST  = cfg["general"]["seed_list"]
PREFIX     = cfg["general"]["prefix"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================================
# 1. Seed Fix
# ===============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===============================================================
# 2. Load Dataset
# ===============================================================
def clean_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)


train_df = pd.read_csv(TRAIN_FILE)
dev_df   = pd.read_csv(DEV_FILE)

train_df["dialogue"] = train_df["dialogue"].astype(str)
train_df["summary"]  = train_df["summary"].apply(clean_text)
dev_df["dialogue"]   = dev_df["dialogue"].astype(str)
dev_df["summary"]    = dev_df["summary"].apply(clean_text)

train_dataset = Dataset.from_pandas(train_df)
dev_dataset   = Dataset.from_pandas(dev_df)


# ===============================================================
# 3. Tokenizer + Special Tokens
# ===============================================================
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)

ENC_MAX = cfg["tokenizer"]["encoder_max_len"]
DEC_MAX = cfg["tokenizer"]["decoder_max_len"]

# ★ 전처리에서 사용한 Person 토큰을 tokenizer에 반드시 등록
special_tokens = cfg["tokenizer"].get("special_tokens", [])
if special_tokens:
    added = tokenizer.add_tokens(special_tokens)
    print(f"[INFO] Added {added} special tokens to tokenizer.")


def preprocess(batch):
    inputs  = [PREFIX + text for text in batch["dialogue"]]
    outputs = [str(o) for o in batch["summary"]]

    enc = tokenizer(
        inputs,
        truncation=True,
        max_length=ENC_MAX,
        padding="max_length",
    )

    dec = tokenizer(
        outputs,
        truncation=True,
        max_length=DEC_MAX,
        padding="max_length",
    )

    labels = dec["input_ids"]
    labels = [
        [-100 if t == tokenizer.pad_token_id else t for t in seq]
        for seq in labels
    ]

    enc["labels"] = labels
    return enc


train_tokenized = train_dataset.map(preprocess, batched=True)
dev_tokenized   = dev_dataset.map(preprocess, batched=True)


# ===============================================================
# 4. ROUGE Metric
# ===============================================================
def compute_rouge(eval_preds):
    preds, labels = eval_preds

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(p, l)["rougeL"].fmeasure for p, l in zip(preds, labels)]
    return {"rougeL": float(np.mean(scores))}


# ===============================================================
# 5. Optuna Objective
# ===============================================================
def objective(trial, cfg, fixed_seed: int):

    hp = cfg["optuna"]["search_space"]

    lr           = trial.suggest_float("learning_rate", float(hp["learning_rate"][0]), float(hp["learning_rate"][1]))
    lora_r       = trial.suggest_categorical("lora_r", hp["lora_r"])
    lora_alpha   = trial.suggest_categorical("lora_alpha", hp["lora_alpha"])
    lora_dropout = trial.suggest_categorical("lora_dropout", hp["lora_dropout"])
    warmup_ratio = trial.suggest_categorical("warmup_ratio", hp["warmup_ratio"])
    epochs       = trial.suggest_categorical("num_train_epochs", hp["num_train_epochs"])

    # ======================================
    # Seed 확실히 고정 + 캐시 정리
    # ======================================
    set_seed(fixed_seed)
    torch.cuda.empty_cache()

    # ======================================
    # W&B init
    # ======================================
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=f"v3_seed_{fixed_seed}_trial_{trial.number}",
        mode=cfg["wandb"]["mode"],
        reinit=True,
        config={
            "seed": fixed_seed,
            "trial": trial.number,
            "learning_rate": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "warmup_ratio": warmup_ratio,
            "num_train_epochs": epochs,
        },
    )

    # ======================================
    # 모델 + LoRA
    # ======================================
    print(f"[INFO] Load base model: {MODEL_NAME}")
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # special tokens 추가에 따른 vocab size 맞추기
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg["training"].get("use_lora", True):
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=cfg["training"]["lora_target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        print("[INFO] LoRA applied.")
    else:
        print("[INFO] use_lora = False → base model only.")

    trial_dir = os.path.join(OUTPUT_DIR, f"seed_{fixed_seed}", f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # generation_max_length은 config와 decoder_max_len 일치시킴
    gen_max_len = cfg["training"].get("generation_max_length", DEC_MAX)

    training_args = Seq2SeqTrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],

        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        optim=cfg["training"]["optim"],

        evaluation_strategy="epoch",
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],

        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        predict_with_generate=cfg["training"]["predict_with_generate"],
        generation_max_length=gen_max_len,
        metric_for_best_model="rougeL",
        greater_is_better=True,

        logging_steps=cfg["training"]["logging_steps"],
        report_to=["wandb"],

        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge,
    )

    trainer.train()

    score = trainer.evaluate()["eval_rougeL"]
    wandb.log({"eval/rougeL": score})
    wandb.finish()

    trial.set_user_attr("score", score)
    print(f"[INFO] Seed {fixed_seed} | Trial {trial.number} | rougeL = {score:.4f}")

    # 메모리 정리
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return score


# ===============================================================
# 6. Run Optuna for each seed
# ===============================================================
def run_seeds(cfg):

    for seed in SEED_LIST:
        print("\n" + "=" * 50)
        print(f"🔥 Starting Optuna for SEED {seed}")
        print("=" * 50 + "\n")

        study = optuna.create_study(direction=cfg["optuna"]["direction"])

        study.optimize(
            lambda t: objective(t, cfg, fixed_seed=seed),
            n_trials=cfg["optuna"]["n_trials"],
        )

        # 결과 저장
        df = pd.DataFrame([
            {
                "seed": seed,
                "trial": t.number,
                "score": t.user_attrs.get("score"),
                "params": t.params,
            }
            for t in study.trials
        ])

        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        out_csv = os.path.join(seed_dir, "trial_scores.csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved trial scores → {out_csv}")

        print(f"\n=== Seed {seed} Optuna 완료 ===")
        print("Best score:", study.best_value)


if __name__ == "__main__":
    run_seeds(cfg)
