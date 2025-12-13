# v5_inference.py — KoBART Dialogue Summarization Inference

import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

from v5_dataset import split_utterances, clean_text


def load_config(path: str = "v5_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_summary(model, tokenizer, dialogue: str, fin_cfg, inf_cfg, noise_cfg):
    # train과 동일한 방식으로 utterance 분리
    utts = split_utterances(dialogue, noise_cfg)

    bos = tokenizer.bos_token or "<s>"
    eos = tokenizer.eos_token or "</s>"
    sep = tokenizer.sep_token or eos

    input_text = bos + sep.join(utts) + eos

    inputs = tokenizer(
        input_text,
        max_length=fin_cfg["max_input_len"],   # encoder 길이
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=inf_cfg["max_target_len"],   # decoder 길이
        num_beams=inf_cfg["num_beams"],
        length_penalty=inf_cfg["length_penalty"],
        use_cache=True,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    cfg = load_config()
    general = cfg["general"]
    fin_cfg = cfg["finetune"]
    inf_cfg = cfg["inference"]
    noise_cfg = cfg["noise"]

    test_path = os.path.join(general["data_dir"], general["test_file"])
    test_df = pd.read_csv(test_path)

    # ---- tokenizer/model: finetuned dir에서 로드 ----
    model_dir = os.path.join(general["output_dir"], f"finetune_{fin_cfg['method']}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Finetuned model dir not found: {model_dir}")

    print(f"[+] Load tokenizer & model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token

    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model.to(inf_cfg["device"])
    model.eval()
    torch.set_grad_enabled(False)

    # ---- Inference ----
    preds = []
    print("[+] Start inference...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row["dialogue"]
        pred = generate_summary(model, tokenizer, dialogue, fin_cfg, inf_cfg, noise_cfg)
        preds.append(pred)

    out_df = pd.DataFrame(
        {
            "fname": test_df["fname"],
            "summary": preds,
        }
    )

    save_path = os.path.join(general["output_dir"], "submission.csv")
    out_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[+] Saved submission to {save_path}")


if __name__ == "__main__":
    main()
