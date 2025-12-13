# v5_train.py — Finetune with (optional) R3F
import os
import random
import numpy as np
import yaml

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
import wandb

from v5_dataset import SummarizationDataset


def load_config(path="v5_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class R3FTrainer(Trainer):
    """
    Trainer with optional R3F regularization.
    method = "default"  -> r3f_lambda = 0 → 일반 CE
    method = "r3f"      -> r3f_lambda > 0
    """

    def __init__(self, r3f_lambda: float = 0.0, noise_std: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.r3f_lambda = r3f_lambda
        self.noise_std = noise_std

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

        pad_token_id = model.config.pad_token_id
        vocab_size = logits.size(-1)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        ce_loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))

        if self.r3f_lambda is None or self.r3f_lambda <= 0.0:
            loss = ce_loss
        else:
            # R3F: 입력 embedding에 작은 노이즈 추가
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            inputs_embeds = model.model.shared(input_ids)
            noise = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=inputs_embeds.size(),
                device=inputs_embeds.device,
            )
            noise_outputs = model(
                inputs_embeds=inputs_embeds + noise,
                attention_mask=attention_mask,
            )
            noise_logits = noise_outputs.logits

            p = logits.view(-1, vocab_size)
            q = noise_logits.view(-1, vocab_size)

            p_log = torch.log_softmax(p, dim=-1)
            q_log = torch.log_softmax(q, dim=-1)
            p_soft = torch.softmax(p, dim=-1)
            q_soft = torch.softmax(q, dim=-1)

            kl1 = torch.nn.functional.kl_div(p_log, q_soft, reduction="batchmean")
            kl2 = torch.nn.functional.kl_div(q_log, p_soft, reduction="batchmean")
            symm_kl = (kl1 + kl2) / 2.0

            loss = ce_loss + self.r3f_lambda * symm_kl

        if return_outputs:
            return loss, outputs
        else:
            return loss


def main():
    cfg_raw = load_config()
    fin_cfg = cfg_raw["finetune"]

    # 🔥 여기서 한 번 정리
    fin_cfg["lr"] = to_float(fin_cfg["lr"])
    fin_cfg["min_lr"] = to_float(fin_cfg["min_lr"])
    fin_cfg["warmup_ratio"] = to_float(fin_cfg["warmup_ratio"])


    set_seed(general.get("seed", 42))

    # -------- Tokenizer --------
    tok_dir = os.path.join(general["output_dir"], "tokenizer")
    if os.path.isdir(tok_dir):
        print(f"[+] Load tokenizer from local: {tok_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        print(f"[+] Load tokenizer from base model: {general['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(general["base_model"])

    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token

    # -------- Model (pretrain or base) --------
    if pre_cfg.get("use_pretrain", False):
        pretrain_dir = os.path.join(general["output_dir"], "pretrain")
        if os.path.isdir(pretrain_dir):
            print(f"[+] Load model from pretrained dir: {pretrain_dir}")
            model = BartForConditionalGeneration.from_pretrained(pretrain_dir)
        else:
            print("[!] pretrain.use_pretrain=True 이지만 pretrain 폴더가 없어 base_model 사용")
            model = BartForConditionalGeneration.from_pretrained(general["base_model"])
    else:
        print(f"[+] Load model from base model: {general['base_model']}")
        model = BartForConditionalGeneration.from_pretrained(general["base_model"])

    # dropout override
    if fin_cfg.get("all_dropout") is not None:
        d = fin_cfg["all_dropout"]
        model.config.dropout = d
        model.config.attention_dropout = d
        model.config.activation_dropout = d
        if hasattr(model.config, "classifier_dropout"):
            model.config.classifier_dropout = d

    # -------- Dataset --------
    train_path = os.path.join(general["data_dir"], general["train_file"])
    dev_path = os.path.join(general["data_dir"], general["dev_file"])

    train_ds = SummarizationDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        max_input_len=fin_cfg["max_input_len"],
        max_target_len=fin_cfg["max_target_len"],
        noise_cfg=noise_cfg,
        is_train=True,
    )

    dev_ds = SummarizationDataset(
        csv_path=dev_path,
        tokenizer=tokenizer,
        max_input_len=fin_cfg["max_input_len"],
        max_target_len=fin_cfg["max_target_len"],
        noise_cfg=noise_cfg,
        is_train=True,
    )

    # -------- wandb --------
    run_name = f"v5_finetune_{fin_cfg['method']}"
    wandb.init(
        project="dialogue_summary_v5",
        name=run_name,
        config=cfg,
    )

    # -------- TrainingArguments --------
    out_dir = os.path.join(general["output_dir"], f"finetune_{fin_cfg['method']}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=fin_cfg["epochs"],
        per_device_train_batch_size=fin_cfg["batch_size"],
        per_device_eval_batch_size=fin_cfg["batch_size"],
        learning_rate=fin_cfg["lr"],
        warmup_ratio=fin_cfg["warmup_ratio"],
        weight_decay=0.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        fp16=fp16,
        report_to=["wandb"],
        run_name=run_name,
    )

    r3f_lambda = fin_cfg["r3f_lambda"] if fin_cfg["method"] == "r3f" else 0.0

    trainer = R3FTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        r3f_lambda=r3f_lambda,
    )

    print("[+] Start finetuning...")
    trainer.train()

    print(f"[+] Save finetuned model to {out_dir}")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
