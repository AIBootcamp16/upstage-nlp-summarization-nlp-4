# ===============================================================
# v4_inference.py — KoBART-base-v2 (No LoRA) + Softmax Weighted Ensemble
#   - train.py와 decoding 설정을 완전히 분리 (정해진 inference 규칙만 사용)
#   - inference는 max_new_tokens 기반 3문장 제한
# ===============================================================

import os
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


# ---------------------------------------------------------------
# 0. Load config
# ---------------------------------------------------------------
def load_config(path="v4_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

DATA_DIR = Path(cfg["general"]["data_dir"])
TEST_FILE = DATA_DIR / cfg["general"]["test_file"]
MODEL_NAME = cfg["general"]["model_name"]
OUTPUT_DIR = Path(cfg["general"]["output_dir"])
PREFIX = cfg["general"]["prefix"]

SEED_LIST = cfg["general"]["seed_list"]

# inference config만 참조
INFER_CFG = cfg["inference"]
MAX_NEW = INFER_CFG["max_new_tokens"]         # 40
NO_REPEAT = INFER_CFG["no_repeat_ngram_size"] # 3
NUM_BEAMS = INFER_CFG["num_beams"]            # 1
LENGTH_PEN = INFER_CFG["length_penalty"]      # 2.0
BATCH_SIZE = INFER_CFG.get("batch_size", 4)

REMOVE_TOKENS = set(INFER_CFG.get("remove_tokens", []))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)


# ---------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)

special_tokens = cfg["tokenizer"]["special_tokens"]
tokenizer.add_tokens(special_tokens)

PAD_ID = tokenizer.pad_token_id
EOS_ID = tokenizer.eos_token_id
ENC_MAX = cfg["tokenizer"]["encoder_max_len"]


# ---------------------------------------------------------------
# 2. Best checkpoint finder
# ---------------------------------------------------------------
def find_best_checkpoint(seed: int):
    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    score_path = seed_dir / "trial_scores.csv"

    if not score_path.exists():
        raise FileNotFoundError()

    df = pd.read_csv(score_path).sort_values("score", ascending=False)
    best_trial = int(df.iloc[0]["trial"])
    best_score = float(df.iloc[0]["score"])

    trial_dir = seed_dir / f"trial_{best_trial}"

    trainer_state = trial_dir / "trainer_state.json"
    if trainer_state.exists():
        state = json.load(open(trainer_state))
        best_ckpt = state.get("best_model_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            return Path(best_ckpt), best_score

    ckpts = sorted(
        list(trial_dir.glob("checkpoint-*")),
        key=lambda x: int(x.name.split("-")[-1]),
        reverse=True
    )
    return ckpts[0], best_score


# ---------------------------------------------------------------
# 3. Load Model (inference 전용 decoding 설정)
# ---------------------------------------------------------------
def load_model(ckpt: Path):
    print(f"[INFO] Loading model from {ckpt}")

    model = BartForConditionalGeneration.from_pretrained(ckpt)
    model.resize_token_embeddings(len(tokenizer))

    model.config.decoder_start_token_id = tokenizer.eos_token_id

    # 🔥 inference 규칙만 적용 (train 설정은 절대 쓰지 않음)
    model.config.num_beams = NUM_BEAMS
    model.config.no_repeat_ngram_size = NO_REPEAT
    model.config.length_penalty = LENGTH_PEN
    model.config.max_new_tokens = MAX_NEW

    model.eval().to(device)
    return model


# ---------------------------------------------------------------
# 4. Greedy Ensemble Decoding (max_new_tokens 기반)
# ---------------------------------------------------------------
@torch.no_grad()
def ensemble_generate(models, weights, dialogues):
    results = []

    for idx in range(0, len(dialogues), BATCH_SIZE):
        batch = dialogues[idx: idx + BATCH_SIZE]
        enc_inputs = [PREFIX + d for d in batch]

        enc = tokenizer(
            enc_inputs,
            truncation=True,
            max_length=ENC_MAX,
            padding=True,
            return_tensors="pt"
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}

        batch_size = enc["input_ids"].size(0)

        cur = torch.full((batch_size, 1), EOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(MAX_NEW):
            fused = None

            for model, w in zip(models, weights):
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=cur,
                )
                logits = out.logits[:, -1, :]

                fused = logits * w if fused is None else fused + logits * w

            # no-repeat-ngram
            if NO_REPEAT > 0:
                for b in range(batch_size):
                    seq = cur[b].tolist()
                    if len(seq) >= NO_REPEAT:
                        prefix = seq[-(NO_REPEAT - 1):]
                        for v in range(fused.size(-1)):
                            test_seq = prefix + [v]
                            if test_seq in [seq[i:i+NO_REPEAT] for i in range(len(seq)-NO_REPEAT+1)]:
                                fused[b, v] = -1e9

            next_token = fused.argmax(dim=-1)
            finished |= (next_token == EOS_ID)
            next_token = next_token.masked_fill(finished, PAD_ID)

            cur = torch.cat([cur, next_token.unsqueeze(-1)], dim=-1)
            if finished.all():
                break

        # decode
        for seq in cur:
            seq = seq.tolist()

            if seq[0] == EOS_ID:
                seq = seq[1:]
            if EOS_ID in seq:
                seq = seq[:seq.index(EOS_ID)]

            text = tokenizer.decode(seq, skip_special_tokens=True)

            for tok in REMOVE_TOKENS:
                text = text.replace(tok, "")

            results.append(text.strip())

    return results


# ---------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------
def main():
    print("[INFO] Load TEST ...")
    test_df = pd.read_csv(TEST_FILE)

    dialogues = test_df["dialogue"].astype(str).tolist()
    fnames = test_df["fname"].tolist()

    models = []
    scores = []

    for seed in SEED_LIST:
        ckpt, score = find_best_checkpoint(seed)
        models.append(load_model(ckpt))
        scores.append(score)

    weights = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)

    preds = ensemble_generate(models, weights, dialogues)

    out_path = OUTPUT_DIR / "v4_submission.csv"
    pd.DataFrame({"fname": fnames, "summary": preds}).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved submission → {out_path}")


if __name__ == "__main__":
    main()
