# v5_dataset.py
import re
import csv
import random
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ==============================
# 1. Noise Cleaning Functions
# ==============================

def clean_text(s: str, noise_cfg: Optional[Dict] = None) -> str:
    """
    noise_cfg:
      - normalize_newline
      - normalize_br
      - normalize_speaker
      - clean_punctuation (지금은 과도한 공백 정도만)
      - strip_whitespace
    """
    if s is None:
        return ""

    cfg = noise_cfg or {}

    # "\\n" -> "\n"
    if cfg.get("normalize_newline", True):
        s = s.replace("\\n", "\n")

    # <br> -> "\n"
    if cfg.get("normalize_br", True):
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)

    # #Person1#, #person1# 등 통일
    if cfg.get("normalize_speaker", True):
        # 콜론 유무 / 공백 유무 다 커버
        s = re.sub(r"#\s*person\s*1\s*#\s*:?", "#Person1#: ", s, flags=re.IGNORECASE)
        s = re.sub(r"#\s*person\s*2\s*#\s*:?", "#Person2#: ", s, flags=re.IGNORECASE)

    # 여러 공백 정리
    if cfg.get("clean_punctuation", True) or cfg.get("strip_whitespace", True):
        s = re.sub(r"[ \t]+", " ", s)

    if cfg.get("strip_whitespace", True):
        s = s.strip()

    return s


def split_utterances(dialogue: str, noise_cfg: Optional[Dict] = None) -> List[str]:
    """
    '#Person1#:' / '#Person2#:' 기준으로 발화 분리.
    만약 speaker 태그가 아예 없으면 '\n' 기준으로 fallback.
    """
    dialogue = clean_text(dialogue, noise_cfg)

    # speaker 태그 기준으로 split
    parts = re.split(r"#Person[12]#:\s*", dialogue)
    utts = [p.strip() for p in parts if p.strip()]

    # speaker 태그가 전혀 없었던 경우
    if len(utts) == 0:
        utts = [u.strip() for u in dialogue.split("\n") if u.strip()]

    return utts


# ==============================
# 2. SummarizationDataset
# ==============================

class SummarizationDataset(Dataset):
    """
    train/dev용 요약 데이터셋
    - input: [BOS] u1 [SEP] u2 [SEP] ... [EOS]
    - labels: 요약문 [BOS] y ... [EOS] (pad는 -100으로 마스킹)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int = 256,
        max_target_len: int = 64,
        noise_cfg: Optional[Dict] = None,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.noise_cfg = noise_cfg
        self.is_train = is_train

        self.items = self._load_csv(csv_path)

    def _load_csv(self, path: str):
        items = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue = row["dialogue"]
                summary = row.get("summary", None)
                fname = row.get("fname", None)

                utts = split_utterances(dialogue, self.noise_cfg)

                items.append(
                    {
                        "fname": fname,
                        "utts": utts,
                        "summary": summary,
                    }
                )
        return items

    def __len__(self):
        return len(self.items)

    def _encode_dialogue(self, utts: List[str]):
        bos = self.tokenizer.bos_token or "<s>"
        eos = self.tokenizer.eos_token or "</s>"
        sep = self.tokenizer.sep_token or eos

        text = bos + sep.join(utts) + eos

        encoded = self.tokenizer(
            text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def _encode_summary(self, summary: Optional[str]):
        if summary is None:
            return None

        bos = self.tokenizer.bos_token or "<s>"
        eos = self.tokenizer.eos_token or "</s>"

        text = bos + summary + eos

        encoded = self.tokenizer(
            text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].squeeze(0)

        # pad 토큰은 -100으로 바꿔서 loss에서 무시
        pad_id = self.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_id] = -100

        return labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        input_ids, attention_mask = self._encode_dialogue(item["utts"])

        if self.is_train:
            labels = self._encode_summary(item["summary"])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            # test/inference용
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


# ==============================
# 3. PretrainDataset (Denoising)
# ==============================

class PretrainDataset(Dataset):
    """
    KoBART Denoising Pretraining:
      - utterance permutation
      - token masking (text infilling)
    input_ids  : 노이즈가 섞인 시퀀스
    labels     : 원본 시퀀스 (pad는 -100)
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
        masking_ratio: float = 0.3,
        noise_cfg: Optional[Dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.masking_ratio = masking_ratio
        self.noise_cfg = noise_cfg

        self.mask_token_id = tokenizer.mask_token_id
        if self.mask_token_id is None:
            # 안전장치: mask_token 없으면 eos를 대신 사용 (실제로는 KoBART에 mask 있음)
            self.mask_token_id = tokenizer.eos_token_id

        self.utts_list = self._load_csv(csv_path)

    def _load_csv(self, path: str):
        data = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue = row["dialogue"]
                utts = split_utterances(dialogue, self.noise_cfg)
                if len(utts) > 0:
                    data.append(utts)
        return data

    def __len__(self):
        return len(self.utts_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        utts = self.utts_list[idx].copy()

        # === Sentence Permutation ===
        random.shuffle(utts)

        bos = self.tokenizer.bos_token or "<s>"
        eos = self.tokenizer.eos_token or "</s>"
        sep = self.tokenizer.sep_token or eos

        text = bos + sep.join(utts) + eos

        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # === Text Infilling (token masking) ===
        valid_positions = torch.where(attention_mask == 1)[0]
        num_tokens = valid_positions.numel()
        num_mask = max(1, int(num_tokens * self.masking_ratio))

        masked_positions = random.sample(valid_positions.tolist(), num_mask)

        corrupted_ids = ids.clone()
        for pos in masked_positions:
            corrupted_ids[pos] = self.mask_token_id

        # labels: 원본 시퀀스, pad는 -100
        labels = ids.clone()
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "input_ids": corrupted_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
