# v5_tokenizer.py
import os
import yaml
from transformers import AutoTokenizer


def load_config(path: str = "v5_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "v5_config.yaml"))
    general = cfg["general"]
    tok_cfg = cfg["tokenizer"]

    if tok_cfg.get("use_pretrained", True):
        src = general["base_model"]
    else:
        src = tok_cfg["tokenizer_path"]

    save_dir = os.path.join(general["output_dir"], "tokenizer")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[+] Loading tokenizer from: {src}")
    tokenizer = AutoTokenizer.from_pretrained(src)

    # sep_token 없으면 eos 재사용 (vocab 안 늘림)
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token

    print(f"[+] Saving tokenizer to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    print("[+] Done! Local tokenizer saved.")


if __name__ == "__main__":
    main()
