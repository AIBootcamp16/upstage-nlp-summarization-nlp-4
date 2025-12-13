# ===============================================================
# v3_inference.py — KoBART + LoRA Multi-Seed Ensemble (FINAL FIX)
# ===============================================================

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from peft import PeftModel
from typing import List


# ===============================================================
# 0. Config Load
# ===============================================================

def load_config(path: str = "v3_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

DATA_DIR   = Path(config["general"]["data_dir"])
TEST_FILE  = DATA_DIR / config["general"]["test_file"]
MODEL_NAME = config["general"]["model_name"]
OUTPUT_DIR = Path(config["general"]["output_dir"])
PREFIX     = config["general"]["prefix"]

SEED_LIST  = config["general"]["seed_list"]

INFER_CFG  = config["inference"]
NUM_BEAMS  = INFER_CFG["num_beams"]
MAX_LENGTH = INFER_CFG["max_length"]
NO_REPEAT_NGRAM_SIZE = INFER_CFG["no_repeat_ngram_size"]
BATCH_SIZE = INFER_CFG["batch_size"]
REMOVE_TOKENS = set(INFER_CFG.get("remove_tokens", []))

TOP_K = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device = {device}")


# ===============================================================
# 1. checkpoint finder
# ===============================================================

def get_best_checkpoint(trial_dir: Path) -> Path:
    state_path = trial_dir / "trainer_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)

        best_raw = state.get("best_model_checkpoint", None)
        if best_raw is not None:
            best_path = Path(best_raw)
            if best_path.exists():
                return best_path
            print(f"[WARN] best_model_checkpoint mismatch → fallback scan: {trial_dir}")

    ckpts = [p for p in trial_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found in {trial_dir}")

    def _step(p: Path):
        try:
            return int(p.name.split("-")[-1])
        except:
            return -1

    return sorted(ckpts, key=_step, reverse=True)[0]


# ===============================================================
# 2. Load tokenizer (+ special tokens 반영)
# ===============================================================

print("[INFO] Load tokenizer...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)

# ★ FIX: 학습에서 사용한 special tokens 강제 로딩
special_tokens = config["tokenizer"].get("special_tokens", [])
if special_tokens:
    added = tokenizer.add_tokens(special_tokens)
    print(f"[INFO] Added {added} special tokens.")

PAD_ID = tokenizer.pad_token_id
EOS_ID = tokenizer.eos_token_id


# ===============================================================
# 3. Load LoRA Model — vocab resize 포함
# ===============================================================

def load_lora_model(best_ckpt: Path) -> torch.nn.Module:
    print(f"[INFO] Load base: {MODEL_NAME}")
    base_model = BartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # ★ vocab resize (학습과 동일하게 맞춤)
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"[INFO] Attach LoRA: {best_ckpt}")
    model = PeftModel.from_pretrained(base_model, best_ckpt)

    if torch.cuda.is_available():
        model = model.to(device).half()
    else:
        model = model.to(device)

    model.eval()
    return model


# ===============================================================
# 4. Text Generation
# ===============================================================

@torch.no_grad()
def generate_summaries(model, dialogues, prefix: str = PREFIX):
    results = []

    for i in range(0, len(dialogues), BATCH_SIZE):
        batch = dialogues[i: i + BATCH_SIZE]
        inputs = [prefix + d for d in batch]

        enc = tokenizer(
            inputs,
            max_length=config["tokenizer"]["encoder_max_len"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        if "token_type_ids" in enc:
            enc.pop("token_type_ids")

        enc = enc.to(device)

        # ★ FIX: pad_token_id / eos_token_id 명시
        gen_ids = model.generate(
            **enc,
            num_beams=NUM_BEAMS,
            max_length=MAX_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
        )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        cleaned = []
        for t in decoded:
            x = t
            for tok in REMOVE_TOKENS:
                x = x.replace(tok, "")
            cleaned.append(x.strip())

        results.extend(cleaned)

    return results


# ===============================================================
# 5. Load ALL Scores → 글로벌 Top-K
# ===============================================================

def load_all_trial_scores() -> pd.DataFrame:
    rows = []

    for seed in SEED_LIST:
        csv_path = OUTPUT_DIR / f"seed_{seed}" / "trial_scores.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["seed"] = seed
        rows.append(df)

    if not rows:
        raise FileNotFoundError("trial_scores.csv not found.")

    full = pd.concat(rows, ignore_index=True)
    full = full.sort_values("score", ascending=False).reset_index(drop=True)

    print(full.head(TOP_K))
    return full.head(TOP_K)


# ===============================================================
# 6. Inference Main
# ===============================================================

def main():
    print("[INFO] Load test.csv ...")
    test_df = pd.read_csv(TEST_FILE)

    fnames = test_df["fname"].tolist()
    dialogues = test_df["dialogue"].astype(str).tolist()

    top_trials = load_all_trial_scores()
    ensemble_out = {"fname": fnames}

    # Best
    best = top_trials.iloc[0]
    best_seed = int(best["seed"])
    best_trial = int(best["trial"])

    for _, row in top_trials.iterrows():
        seed = int(row["seed"])
        trial = int(row["trial"])

        trial_dir = OUTPUT_DIR / f"seed_{seed}" / f"trial_{trial}"
        best_ckpt = get_best_checkpoint(trial_dir)

        print(f"[INFO] Infer: seed={seed}, trial={trial}")
        model = load_lora_model(best_ckpt)

        preds = generate_summaries(model, dialogues)
        col_name = f"summary_seed{seed}_trial{trial}"
        ensemble_out[col_name] = preds

        del model
        torch.cuda.empty_cache()

    # Save best
    best_col = f"summary_seed{best_seed}_trial{best_trial}"
    submission = pd.DataFrame({"fname": fnames, "summary": ensemble_out[best_col]})
    submission.to_csv(OUTPUT_DIR / "v3_submission_best.csv", index=False, encoding="utf-8-sig")

    # Save candidates
    pd.DataFrame(ensemble_out).to_csv(
        OUTPUT_DIR / "v3_candidate_summaries.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[INFO] Inference complete!")


if __name__ == "__main__":
    main()
