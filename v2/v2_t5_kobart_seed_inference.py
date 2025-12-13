# ===============================================================
# v2_t5_kobart_seed_inference.py — FINAL STABLE VERSION (REAL)
# ===============================================================

import os
import gc
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Config Loader
# -------------------------
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Test Loader
# -------------------------
def load_test_dataset(cfg):
    test_path = os.path.join(cfg["general"]["data_dir"], cfg["general"]["test_file"])
    df = pd.read_csv(test_path)

    df = df.rename(columns={"dialogue_clean": "input_text"})
    df["len"] = df["input_text"].apply(lambda x: len(str(x).split()))
    return df


# ===============================================================
# KoBART: seed_* 디렉토리 찾기
# ===============================================================
def list_seed_dirs(base_dir: str):
    if not os.path.exists(base_dir):
        raise RuntimeError(f"❌ base_dir not found: {base_dir}")

    seeds = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("seed_"):
            seeds.append(full)

    if not seeds:
        raise RuntimeError(f"❌ No seed_* dirs inside {base_dir}")

    print(f"[INFO] KoBART seeds under {base_dir}:")
    for s in seeds:
        print("   -", s)

    return sorted(seeds)


# ===============================================================
# T5: trial_* 디렉토리 찾기
# ===============================================================
def list_trial_dirs(base_dir: str):
    if not os.path.exists(base_dir):
        raise RuntimeError(f"❌ base_dir not found: {base_dir}")

    trials = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("trial_"):
            trials.append(full)

    if not trials:
        raise RuntimeError(f"❌ No trial_* dirs inside {base_dir}")

    print(f"[INFO] T5 trials under {base_dir}:")
    for t in trials:
        print("   -", t)

    return sorted(trials)


# ===============================================================
# checkpoint-* 자동 탐색
# ===============================================================
def find_best_checkpoint(run_dir: str):

    ckpts = []
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            ckpts.append(full)

    if not ckpts:
        raise RuntimeError(
            f"❌ checkpoint-* NOT FOUND under {run_dir}. "
            "Training did not save checkpoints."
        )

    ckpts = sorted(
        ckpts,
        key=lambda x: int(os.path.basename(x).split("-")[-1])
    )

    best = ckpts[-1]
    print(f"[CHECKPOINT] Selected: {best}")
    return best


# ===============================================================
# 모델 로더 (safetensors 지원)
# ===============================================================
def load_model_and_tokenizer(cfg, ckpt_path):
    print(f"[LOAD] Loading checkpoint: {ckpt_path}")

    # ✅ 1) tokenizer도 checkpoint에서 직접 로딩
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # ❌ inference 단계에서는 special_tokens 다시 add 하면 안 됨
    # special = cfg.get("tokenizer", {}).get("special_tokens")
    # if special:
    #     tokenizer.add_tokens(special)

    # ✅ 2) model도 checkpoint에서 로딩 (safetensors 자동 인식)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
    )

    # ❌ 여기서 resize_token_embeddings도 하면 안 됨
    # if special:
    #     model.resize_token_embeddings(len(tokenizer))

    if DEVICE == "cuda":
        model = model.to(DEVICE).half()

    model.eval()
    return tokenizer, model


# ===============================================================
# Generate
# ===============================================================
@torch.no_grad()
def generate_summaries(model, tokenizer, df, cfg, tag):
    bs = min(cfg["inference"]["batch_size"], 4)
    prefix = cfg["general"].get("prefix", "")

    results = []

    for i in tqdm(range(0, len(df), bs), desc=f"{tag} inference"):
        texts = df["input_text"].iloc[i:i+bs].tolist()
        texts = [prefix + x for x in texts]

        # 1) tokenizer
        enc = tokenizer(
            texts,
            max_length=cfg["tokenizer"]["encoder_max_len"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # 2) token_type_ids 무조건 제거
        enc = {k: v for k, v in enc.items() if k != "token_type_ids"}

        # 3) DEVICE 이동
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # 4) generate 호출 시 token_type_ids 강제 차단
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=cfg["inference"]["max_length"],
                num_beams=cfg["inference"]["num_beams"],
                no_repeat_ngram_size=cfg["inference"]["no_repeat_ngram_size"],
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)

        del enc, out
        torch.cuda.empty_cache()

    return results


# ===============================================================
# Main
# ===============================================================
def run_inference(cfg_t5, cfg_kb):

    df = load_test_dataset(cfg_t5)

    short_df = df[df["len"] <= 100].copy()
    long_df  = df[df["len"] >  100].copy()

    print(f"SHORT: {len(short_df)} rows")
    print(f"LONG : {len(long_df)} rows\n")

    # --- KoBART seed dirs ---
    kb_seeds = list_seed_dirs(cfg_kb["general"]["output_dir"])

    # --- T5 trial dirs ---
    t5_trials = list_trial_dirs(cfg_t5["general"]["output_dir"])

    # ----------------------
    # SHORT → KoBART
    # ----------------------
    short_pred = {}

    for sd in kb_seeds:
        ckpt = find_best_checkpoint(sd)
        tok, mod = load_model_and_tokenizer(cfg_kb, ckpt)
        short_pred[sd] = generate_summaries(mod, tok, short_df, cfg_kb, "KoBART")

        del tok, mod
        gc.collect()
        torch.cuda.empty_cache()

    if len(short_df) > 0:
        final = []
        for i in range(len(short_df)):
            cand = [short_pred[s][i] for s in kb_seeds]
            lens = [len(c) for c in cand]
            avg = sum(lens) / len(lens)
            best = min(range(len(cand)), key=lambda j: abs(len(cand[j]) - avg))
            final.append(cand[best])
        short_df["summary"] = final

    # ----------------------
    # LONG → T5
    # ----------------------
    long_pred = {}

    for tr in t5_trials:
        ckpt = find_best_checkpoint(tr)
        tok, mod = load_model_and_tokenizer(cfg_t5, ckpt)
        long_pred[tr] = generate_summaries(mod, tok, long_df, cfg_t5, "T5")

        del tok, mod
        gc.collect()
        torch.cuda.empty_cache()

    if len(long_df) > 0:
        final = []
        for i in range(len(long_df)):
            cand = [long_pred[t][i] for t in t5_trials]
            lens = [len(c) for c in cand]
            avg = sum(lens) / len(lens)
            best = min(range(len(cand)), key=lambda j: abs(len(cand[j]) - avg))
            final.append(cand[best])
        long_df["summary"] = final

    # SAVE
    out = pd.concat([short_df, long_df]).sort_index()
    out[["fname", "summary"]].to_csv("v2_t5_kobart_seed_submission_final.csv", index=False)

    print("\n🎉 Saved → v2_t5_kobart_seed_submission_final.csv")


# ===============================================================
# Entry
# ===============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config_t5", required=True)
    p.add_argument("--config_kobart", required=True)
    args = p.parse_args()

    cfg_t5 = load_config(args.config_t5)
    cfg_kb = load_config(args.config_kobart)

    run_inference(cfg_t5, cfg_kb)
