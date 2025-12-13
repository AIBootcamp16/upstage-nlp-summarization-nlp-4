
# ===============================================================
# v4_no_lora_train.py — KoBART Full Fine-Tune (No LoRA)
#  - special tokens 완전 학습
#  - v4 전처리와 100% 호환
#  - decoder_start_token_id 명시 (KoBART)
#  - seed 완전 고정
#  - train/eval decoding 세팅 ≒ inference 세팅으로 정렬
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
    set_seed as hf_set_seed,
)
from rouge_score import rouge_scorer

torch.backends.cuda.matmul.allow_tf32 = True

# ===============================================================
# 0. Load Config
# ===============================================================
def load_config(path: str = "v4_config.yaml"):
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
# 1. Fix Seed
# ===============================================================
def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

# ===============================================================
# 2. Dataset Load
# ===============================================================
def clean_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)

def normalize_dialogue_column(df: pd.DataFrame) -> pd.DataFrame:
    def _fix(x):
        # Case 1: Python list 그대로 → 공백 join
        if isinstance(x, list):
            return " ".join(map(str, x))

        # Case 2: 문자열인데 리스트처럼 생긴 경우 → eval 후 join
        if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
            try:
                arr = eval(x)
                if isinstance(arr, list):
                    return " ".join(map(str, arr))
            except:
                pass  # 실패하면 그냥 문자열로 처리

        # Case 3: 일반 문자열 → 그대로 사용
        return str(x)

    df["dialogue"] = df["dialogue"].apply(_fix)
    return df

# ===============================================================
# 2. Dataset Load (FINAL, list → str 완전 강제 변환 포함)
# ===============================================================

def load_datasets():
    """
    v4 전처리 파이프라인의 안전성을 위해,
    - dialogue/summary를 무조건 문자열(str)로 강제 변환
    - 리스트처럼 생긴 문자열도 절대 literal_eval 하지 않음
    - flatten / join 처리도 하지 않음 (preprocess_builder에서 처리)
    """

    train_df = pd.read_csv(TRAIN_FILE)
    dev_df   = pd.read_csv(DEV_FILE)

    # --- 모든 입력을 무조건 문자열(str)로 변환 ---
    train_df["dialogue"] = train_df["dialogue"].astype(str)
    dev_df["dialogue"]   = dev_df["dialogue"].astype(str)

    train_df["summary"] = train_df["summary"].astype(str)
    dev_df["summary"]   = dev_df["summary"].astype(str)

    # --- HuggingFace Dataset 변환 ---
    train_ds = Dataset.from_pandas(train_df)
    dev_ds   = Dataset.from_pandas(dev_df)

    # --- map 단계에서는 더 이상 변환 없음 ---
    return train_ds, dev_ds

# ===============================================================
# 3. Tokenizer
# ===============================================================
def build_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)

    special_tokens = cfg["tokenizer"].get("special_tokens", [])
    if special_tokens:
        added = tokenizer.add_tokens(special_tokens)
        print(f"[INFO] Added {added} special tokens:", special_tokens)

    ENC_MAX = cfg["tokenizer"]["encoder_max_len"]
    DEC_MAX = cfg["tokenizer"]["decoder_max_len"]
    return tokenizer, ENC_MAX, DEC_MAX

# ===============================================================
# 4. Preprocess
# ===============================================================
def preprocess_builder(tokenizer, ENC_MAX, DEC_MAX):
    def preprocess(batch):
        import ast

        # 1) dialogue 평탄화
        dialogues = []
        for x in batch["dialogue"]:
            if isinstance(x, list):
                dialogues.append(" ".join(map(str, x)))
            elif isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                try:
                    arr = ast.literal_eval(x)
                    if isinstance(arr, list):
                        dialogues.append(" ".join(map(str, arr)))
                        continue
                except:
                    pass
                dialogues.append(x)
            else:
                dialogues.append(str(x))

        # 2) summary 평탄화
        summaries = []
        for x in batch["summary"]:
            if isinstance(x, list):
                summaries.append(" ".join(map(str, x)))
            else:
                summaries.append(str(x))

        # 3) Tokenize
        inputs = [PREFIX + d for d in dialogues]

        enc = tokenizer(
            inputs,
            truncation=True,
            max_length=ENC_MAX,
            padding="max_length",
        )
        dec = tokenizer(
            summaries,
            truncation=True,
            max_length=DEC_MAX,
            padding="max_length",
        )

        # 4) Labels (-100 masking)
        labels = dec["input_ids"]
        pad_id = tokenizer.pad_token_id
        labels = [
            [-100 if t == pad_id else t for t in seq]
            for seq in labels
        ]
        enc["labels"] = labels

        # 5) KoBART는 token_type_ids 미지원 → 제거
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        if "token_type_ids" in dec:
            del dec["token_type_ids"]

        return enc

    return preprocess

# ===============================================================
# 5. ROUGE Metric
# ===============================================================
from rouge_score import rouge_scorer

def build_rouge_fn(tokenizer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute_rouge(eval_preds):
        preds, labels = eval_preds
        pad_id = tokenizer.pad_token_id

        # -100 → pad_id 로 복구 후 디코딩
        labels = np.where(labels != -100, labels, pad_id)

        preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = [
            scorer.score(p, l)["rougeL"].fmeasure
            for p, l in zip(preds, labels)
        ]
        return {"rougeL": float(np.mean(scores))}
    
    return compute_rouge

# ===============================================================
# 6. Build Model — FULL FINETUNE (No LoRA)
# ===============================================================
# ===============================================================
# 6. Build Model — FULL FINETUNE (No LoRA)
# ===============================================================
def build_model(tokenizer):
    print("[INFO] Load base:", MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # KoBART decoder start token
    model.config.decoder_start_token_id = tokenizer.eos_token_id

    # ============ Decoding 설정 (훈련·검증용) ============
    # → 모두 training 섹션 기준으로만 맞춘다
    GEN_MAX       = cfg["training"]["generation_max_length"]           # 예: 40 or 64
    GEN_BEAMS     = cfg["training"]["generation_num_beams"]            # 예: 1
    GEN_NO_REPEAT = cfg["training"]["generation_no_repeat_ngram_size"] # 예: 3
    

    # Trainer가 eval에서 generate() 호출할 때 참고하는 값들
    model.config.max_length = GEN_MAX
    model.config.num_beams = GEN_BEAMS
    model.config.no_repeat_ngram_size = GEN_NO_REPEAT
    

    # KoBART에 special tokens 반영
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ===============================================================
# 7. Optuna Objective
# ===============================================================
def objective(trial, seed, tokenizer, train_tok, dev_tok):
    hp_cfg = cfg["optuna"]["search_space"]

    hp = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(hp_cfg["learning_rate"][0]),
            float(hp_cfg["learning_rate"][1]),
        ),
        "warmup_ratio": trial.suggest_categorical(
            "warmup_ratio",
            hp_cfg["warmup_ratio"],
        ),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs",
            hp_cfg["num_train_epochs"],
        ),
    }

    fix_seed(seed)
    torch.cuda.empty_cache()

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=f"v4_nolora_seed{seed}_trial{trial.number}",
        mode=cfg["wandb"]["mode"],
        reinit=True,
        config={"seed": seed, "trial": trial.number, **hp},
        group=f"seed_{seed}",
    )

    model = build_model(tokenizer)

    # === train/eval generation 설정 (훈련 안정용 기본값) ===
    GEN_MAX       = cfg["training"]["generation_max_length"]       # 보통 40~64
    GEN_BEAMS     = cfg["training"]["generation_num_beams"]        # 보통 1 또는 4
    GEN_NO_REPEAT = cfg["training"]["generation_no_repeat_ngram_size"]


    out_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}", f"trial_{trial.number}")
    os.makedirs(out_dir, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        seed=seed,

        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],

        num_train_epochs=hp["num_train_epochs"],
        learning_rate=hp["learning_rate"],
        warmup_ratio=hp["warmup_ratio"],

        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        optim=cfg["training"]["optim"],

        evaluation_strategy="epoch",
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,

        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        predict_with_generate=True,

               # 🔥 train/eval에서 쓸 디코딩 세팅 (config.training 기준)
        generation_max_length=cfg["training"]["generation_max_length"],              # ex) 40
        


        metric_for_best_model="rougeL",
        greater_is_better=True,

        logging_steps=cfg["training"]["logging_steps"],
        report_to=["wandb"],
        eval_accumulation_steps=1,
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
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_rouge_fn(tokenizer),
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    score = float(eval_metrics["eval_rougeL"])

    trial.set_user_attr("score", score)

    wandb.log({"eval/rougeL": score})
    wandb.finish()

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return score

# ===============================================================
# 8. Main
# ===============================================================
def run_seeds():
    tokenizer, ENC_MAX, DEC_MAX = build_tokenizer()
    train_dataset, dev_dataset = load_datasets()

    # fname / topic 제거는 Dataset 레벨에서 한 번 더 안전하게
    for col in ["fname", "topic", "__index_level_0__"]:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(col)
        if col in dev_dataset.column_names:
            dev_dataset = dev_dataset.remove_columns(col)

    preprocess = preprocess_builder(tokenizer, ENC_MAX, DEC_MAX)
    train_tok = train_dataset.map(preprocess, batched=True)
    dev_tok   = dev_dataset.map(preprocess, batched=True)

    # 🔥 여기서 원인 제거: 더 이상 필요 없는 원본 텍스트 컬럼 제거
    drop_cols = []
    for c in ["dialogue", "summary"]:
        if c in train_tok.column_names:
            drop_cols.append(c)
    if drop_cols:
        print(f"[INFO] Removing unused text columns from tokenized datasets: {drop_cols}")
        train_tok = train_tok.remove_columns(drop_cols)
        dev_tok   = dev_tok.remove_columns(drop_cols)

    for seed in SEED_LIST:
        print(f"\n======================= SEED {seed} =======================")
        fix_seed(seed)

        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        study = optuna.create_study(direction=cfg["optuna"]["direction"])
        study.optimize(
            lambda t: objective(t, seed, tokenizer, train_tok, dev_tok),
            n_trials=cfg["optuna"]["n_trials"],
        )

        df = pd.DataFrame(
            [
                {
                    "seed": seed,
                    "trial": t.number,
                    "score": t.user_attrs.get("score"),
                    **t.params,
                }
                for t in study.trials
            ]
        )
        out_csv = os.path.join(seed_dir, "trial_scores.csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved trial scores → {out_csv}")
        print(f"[INFO] Best Score (SEED {seed}): {study.best_value:.4f}")


if __name__ == "__main__":
    run_seeds()
