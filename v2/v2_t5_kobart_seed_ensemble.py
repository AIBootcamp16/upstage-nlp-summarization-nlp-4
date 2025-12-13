# ===============================================================
# T5 Optuna + KoBART Seed Ensemble (OOM 완화 안정 버전, 3090 기준)
#   - VSCode / Terminal 실행 전용
#   - 반드시 config_t5 / config_kobart 두 개 받음
# ===============================================================

import os
import gc
import yaml
import wandb
import optuna
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from functools import partial
from rouge_score import rouge_scorer
from transformers import set_seed

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

torch.backends.cuda.matmul.allow_tf32 = True

# 전역 ROUGE scorer (매번 생성하지 않도록)
ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)


# ===============================================================
# 0) Load config
# ===============================================================
def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ===============================================================
# 1) Load dataset
# ===============================================================
def load_train_valid(cfg):
    df_path = os.path.join(cfg["general"]["data_dir"], cfg["general"]["train_file"])
    df = pd.read_csv(df_path)

    df = df[["dialogue_clean", "summary"]].rename(
        columns={"dialogue_clean": "input_text", "summary": "target_text"}
    )

    train_df = df.sample(frac=0.9, random_state=cfg["general"]["seed"])
    valid_df = df.drop(train_df.index)

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(valid_df.reset_index(drop=True)),
        valid_df.reset_index(drop=True),
    )


# ===============================================================
# 2) Tokenizer + Model
# ===============================================================
def create_tokenizer_and_model(cfg):
    model_name = cfg["general"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_added = 0
    if cfg["tokenizer"].get("special_tokens"):
        num_added = tokenizer.add_tokens(cfg["tokenizer"]["special_tokens"])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


# ===============================================================
# 3) Preprocess (dynamic padding 전제)
# ===============================================================
def preprocess(batch, tokenizer, cfg_tokenizer, prefix=""):
    # prefix 붙인 입력 텍스트
    inputs = [prefix + x for x in batch["input_text"]]

    # ⬇ 여기서는 padding 하지 않음 → DataCollator가 batch 단위로 dynamic padding
    model_inputs = tokenizer(
        inputs,
        max_length=cfg_tokenizer["encoder_max_len"],
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target_text"],
            max_length=cfg_tokenizer["decoder_max_len"],
            truncation=True,
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs


# ===============================================================
# 4) ROUGE Metric
# ===============================================================
def compute_rouge_scores(pred, refs):
    r1 = ROUGE_SCORER.score(refs[0], pred)["rouge1"].fmeasure
    r2 = ROUGE_SCORER.score(refs[0], pred)["rouge2"].fmeasure
    rl = ROUGE_SCORER.score(refs[0], pred)["rougeL"].fmeasure
    return float(r1 + r2 + rl)


def compute_metrics(eval_pred, tokenizer, gold_df):
    preds, _ = eval_pred

    # logits → ids
    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:
        preds = preds.argmax(-1)

    preds = np.clip(preds.astype(np.int64), 0, tokenizer.vocab_size - 1)
    decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)

    scores = []
    N = min(len(decoded), len(gold_df))  # 길이 mismatch 방지

    for i in range(N):
        ref = gold_df.iloc[i]["target_text"]
        pred = decoded[i]
        try:
            s = compute_rouge_scores(pred, [ref])
            scores.append(s)
        except Exception:
            continue

    if len(scores) == 0:
        return {"final_rouge": 0.0}

    return {"final_rouge": float(np.mean(scores))}


# ===============================================================
# 5) Single training run (OOM 완화 버전)
# ===============================================================
def train_single_run(
    cfg,
    train_dataset,
    valid_dataset,
    valid_df,
    seed,
    lr,
    warmup,
    epochs,
    run_name_suffix,
    output_subdir,
):

    set_seed(seed)

    tokenizer, model = create_tokenizer_and_model(cfg)
    prefix = cfg["general"].get("prefix", "")

    # dynamic padding 전제 토크나이즈
    tok_train = train_dataset.map(
        partial(
            preprocess,
            tokenizer=tokenizer,
            cfg_tokenizer=cfg["tokenizer"],
            prefix=prefix,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tok_valid = valid_dataset.map(
        partial(
            preprocess,
            tokenizer=tokenizer,
            cfg_tokenizer=cfg["tokenizer"],
            prefix=prefix,
        ),
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    out_dir = os.path.join(cfg["general"]["output_dir"], output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=f"{cfg['wandb']['name']}_{run_name_suffix}",
        mode=cfg["wandb"]["mode"],
        reinit=True,
    )

    # DataCollator: batch 단위로 dynamic padding + label pad = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # eval batch size는 1로 고정(혹은 config 있으면 사용) → 메모리 절약
    per_device_eval_batch_size = cfg["training"].get(
        "per_device_eval_batch_size", 1
    )

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        warmup_ratio=warmup,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        evaluation_strategy="epoch",
        # 🔴 ROUGE 계산 위해 무조건 generate 사용
        predict_with_generate=True,
        fp16=cfg["training"]["fp16"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        report_to=["wandb"],
        seed=seed,
        gradient_checkpointing=True,  # ✅ 메모리 절약
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, valid_df),
    )

    try:
        trainer.train()

        # 🔧 여기서 KeyError 안 나게 안전하게 가져오기
        metrics = trainer.evaluate()
        # HF는 eval_ 접두사 붙이므로 우선적으로 eval_final_rouge 찾기
        score = (
            metrics.get("eval_final_rouge")
            or metrics.get("final_rouge")
            or 0.0
        )

        wandb.log({"final_rouge": score})
    finally:
        wandb.finish()

        # 🔥 GPU / CPU 메모리 강제 정리
        del trainer
        del model
        del tokenizer
        del tok_train
        del tok_valid
        gc.collect()
        torch.cuda.empty_cache()

    return score


# ===============================================================
# 6) T5 Optuna
# ===============================================================
def t5_objective(trial, cfg, train_dataset, valid_dataset, valid_df):
    lr = trial.suggest_categorical(
        "learning_rate", cfg["optuna"]["search_space"]["learning_rate"]
    )
    warm = trial.suggest_categorical(
        "warmup_ratio", cfg["optuna"]["search_space"]["warmup_ratio"]
    )
    epochs = trial.suggest_categorical(
        "num_train_epochs", cfg["optuna"]["search_space"]["num_train_epochs"]
    )

    trial_seed = cfg["general"]["seed"] + trial.number

    return train_single_run(
        cfg,
        train_dataset,
        valid_dataset,
        valid_df,
        seed=trial_seed,
        lr=lr,
        warmup=warm,
        epochs=epochs,
        run_name_suffix=f"t5_trial{trial.number}_seed{trial_seed}",
        output_subdir=f"trial_{trial.number}_seed_{trial_seed}",
    )


# ===============================================================
# 7) Train T5 & KoBART
# ===============================================================
def run_training(
    config_t5,
    config_kobart,
    train_t5,
    valid_t5,
    df_t5,
    train_kb,
    valid_kb,
    df_kb,
):

    print("\n==============================")
    print("🔥 T5 OPTUNA TRAINING START")
    print("==============================")

    # 병렬 trial 금지(n_jobs=1) → 메모리 보호
    study = optuna.create_study(
        direction=config_t5["optuna"]["direction"]
    )
    study.optimize(
        lambda trial: t5_objective(trial, config_t5, train_t5, valid_t5, df_t5),
        n_trials=config_t5["optuna"]["n_trials"],
    )

    print("\n🔥 BEST TRIAL:", study.best_trial.number, study.best_value)

    # T5 끝난 뒤 한 번 더 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    print("\n==============================")
    print("🔥 KOBART SEED ENSEMBLE START")
    print("==============================")

    base_seed = config_kobart["general"]["seed"]
    seeds = config_kobart.get("ensemble", {}).get(
        "seeds", [base_seed, base_seed + 2025]
    )

    for s in seeds:
        _ = train_single_run(
            config_kobart,
            train_kb,
            valid_kb,
            df_kb,
            seed=s,
            lr=config_kobart["training"]["learning_rate"],
            warmup=config_kobart["training"]["warmup_ratio"],
            epochs=config_kobart["training"]["num_train_epochs"],
            run_name_suffix=f"kobart_seed{s}",
            output_subdir=f"seed_{s}",
        )

        # seed 하나 끝날 때마다 정리 (추가 안전장치)
        gc.collect()
        torch.cuda.empty_cache()


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_t5", type=str, required=True)
    parser.add_argument("--config_kobart", type=str, required=True)
    parser.add_argument("--skip_t5", action="store_true")  # 🔥 추가된 옵션
    args = parser.parse_args()

    config_t5 = load_config(args.config_t5)
    config_kobart = load_config(args.config_kobart)

    # ============================================================
    # OPTION: --skip_t5 → T5 Optuna 건너뛰고 KoBART만 실행
    # ============================================================
    if args.skip_t5:
        print("\n==============================")
        print("🔥 SKIP T5 → KOBART ONLY MODE")
        print("==============================")

        train_kb, valid_kb, df_kb = load_train_valid(config_kobart)

        base_seed = config_kobart["general"]["seed"]
        seeds = config_kobart.get("ensemble", {}).get(
            "seeds", [base_seed, base_seed + 2025]
        )

        for s in seeds:
            train_single_run(
                config_kobart,
                train_kb,
                valid_kb,
                df_kb,
                seed=s,
                lr=config_kobart["training"]["learning_rate"],
                warmup=config_kobart["training"]["warmup_ratio"],
                epochs=config_kobart["training"]["num_train_epochs"],
                run_name_suffix=f"kobart_seed{s}",
                output_subdir=f"seed_{s}",
            )
        exit()  # 🔥 코바트 끝나면 종료

    # ============================================================
    # DEFAULT: T5 Optuna + KoBART
    # ============================================================
    train_t5, valid_t5, df_t5 = load_train_valid(config_t5)
    train_kb, valid_kb, df_kb = load_train_valid(config_kobart)

    run_training(
        config_t5,
        config_kobart,
        train_t5,
        valid_t5,
        df_t5,
        train_kb,
        valid_kb,
        df_kb,
    )
