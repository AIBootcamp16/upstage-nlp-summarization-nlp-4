# ============================================
# v5_pretrain.py — KoBART Denoising Pretraining
# ============================================

import os
import random
import numpy as np
import yaml

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import wandb

from v5_dataset import PretrainDataset  # noise_cfg 반영된 최신 버전 사용


# ============================
# Load Config
# ============================
def load_config(path: str = "v5_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================
# Safe float cast
# ============================
def to_float(x):
    try:
        return float(x)
    except Exception:
        return x


# ============================
# Seed
# ============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================
# Main
# ============================
def main():
    # ------------------ Load Config ------------------
    cfg = load_config("v5_config.yaml")

    general_cfg = cfg["general"]
    pre_cfg = cfg["pretrain"]
    tok_cfg = cfg["tokenizer"]
    noise_cfg = cfg["noise"]

    # LR 이상형태 방지: 강제 float 캐스팅
    pre_cfg["lr"] = to_float(pre_cfg["lr"])
    pre_cfg["min_lr"] = to_float(pre_cfg["min_lr"])
    pre_cfg["warmup_ratio"] = to_float(pre_cfg["warmup_ratio"])

    set_seed(general_cfg.get("seed", 42))

    # ------------------ Path 설정 ------------------
    data_dir = general_cfg["data_dir"]
    train_path = os.path.join(data_dir, general_cfg["train_file"])
    dev_path = os.path.join(data_dir, general_cfg["dev_file"])

    output_dir = os.path.join(general_cfg["output_dir"], "pretrain")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------ Tokenizer ------------------
    tok_dir = os.path.join(general_cfg["output_dir"], "tokenizer")

    if os.path.isdir(tok_dir):
        print(f"[+] Load tokenizer from local: {tok_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        print(f"[+] Load tokenizer from pretrained: {general_cfg['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(general_cfg["base_model"])

    # sep_token 없으면 eos 재사용
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token

    # ------------------ Model ------------------
    print(f"[+] Load base model: {general_cfg['base_model']}")
    model = BartForConditionalGeneration.from_pretrained(general_cfg["base_model"])

    # tokenizer 크기와 model embedding 맞추기
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("[+] Resize token embeddings")
        model.resize_token_embeddings(len(tokenizer))

    # ------------------ Dataset ------------------
    print("[+] Load datasets (pretraining mode)...")

    train_ds = PretrainDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        max_seq_len=pre_cfg["max_seq_len"],
        masking_ratio=pre_cfg["masking_ratio"],
        noise_cfg=noise_cfg,
    )
    dev_ds = PretrainDataset(
        csv_path=dev_path,
        tokenizer=tokenizer,
        max_seq_len=pre_cfg["max_seq_len"],
        masking_ratio=pre_cfg["masking_ratio"],
        noise_cfg=noise_cfg,
    )

    # ------------------ Data Collator ------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )

    # ------------------ WandB ------------------
    run_name = "v5_pretrain_kobart"
    wandb.init(
        project="dialogue_summary_v5",
        name=run_name,
        config=cfg,
    )

    # ------------------ Training Arguments ------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (device == "cuda")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=pre_cfg["epochs"],
        per_device_train_batch_size=pre_cfg["batch_size"],
        per_device_eval_batch_size=pre_cfg["batch_size"],
        learning_rate=pre_cfg["lr"],
        warmup_ratio=pre_cfg["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=3,
        fp16=fp16,
        report_to=["wandb"],
        run_name=run_name,
    )

    # ------------------ Trainer ------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ------------------ Train ------------------
    print("[+] Start KoBART pretraining (denoising)...")
    trainer.train()

    print(f"[+] Save final pretrained model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
