<p align="center"> <img src="https://capsule-render.vercel.app/api?type=soft&height=180&text=💬%20Dialogue%20Summarization&fontSize=35&descSize=20&descAlignY=65&color=gradient&customColorList=0,2,6,11,20&fontColor=ffffff&animation=fadeIn" alt="Dialogue Summarization | 일상 대화 요약"/> </p>

<h2 align="center"> 4조 VIBE (Visual Intelligence and Best Engineering) </h2> <h3 align="center"> 🥈 Leaderboard 2nd Place (Final Score: 47.7938) </h3> <h3 align="center"> 📝 일상 대화 텍스트의 맥락을 파악하여 핵심 요약문을 생성 </h3>

<div align="center"> <img src="https://img.shields.io/badge/Python-3.8-blue?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface&logoColor=white"/> <img src="https://img.shields.io/badge/Solar-API-orange?style=flat&logo=openai&logoColor=white"/> </div>

<br>

## 👨‍👩‍👦‍👦 팀 구성원

<div align="center">

<table>
<tr>
<td align="center" width="200px">
<a href="https://github.com/imeanseo"><img src="https://avatars.githubusercontent.com/u/221927853?v=4" width="200px;" alt="고민서"/></a>
</td>
<td align="center" width="200px">
<a href="https://github.com/oriori88"><img src="https://avatars.githubusercontent.com/u/11532528?v=4" width="200px;" alt="김소은"/></a>
</td>
<td align="center" width="200px">
<a href="https://github.com/Leesoomin97"><img src="https://avatars.githubusercontent.com/u/218931464?v=4" width="200px;" alt="이수민"/></a>
</td>
<td align="center" width="200px">
<a href="https://github.com/dg5407"><img src="https://avatars.githubusercontent.com/u/221937194?v=4" width="200px;" alt="이동건"/></a>
</td>
<td align="center" width="200px">
<a href="https://github.com/vforjj"><img src="https://avatars.githubusercontent.com/u/227140598?v=4" width="200px;" alt="주예령"/></a>
</td>
</tr>
<tr>
<td align="center"><a href="https://github.com/imeanseo"><b>👑 고민서 (팀장)</b></a></td>
<td align="center"><a href="https://github.com/oriori88"><b>😺 김소은</b></a></td>
<td align="center"><a href="https://github.com/Leesoomin97"><b>🐿️ 이수민</b></a></td>
<td align="center"><a href="https://github.com/dg5407"><b>👨🏻‍💻 이동건</b></a></td>
<td align="center"><a href="https://github.com/vforjj"><b>🐥 주예령</b></a></td>
</tr>
<tr>
<td align="center">
    <sub><b>📌 담당 역할</b></sub><br>
    <sub>✅ EDA </sub><br>
    <sub>✅ Preprocessing</sub><br>
    <sub>✅ Modeling </sub><br>
    <sub>✅ Organize PPT </sub><br>
    <sub>✅ Organize ReadMe/Git </sub>
</td>
<td align="center">
    <sub><b>📌 담당 역할</b></sub><br>
    <sub>✅ EDA </sub><br>
    <sub>✅ Preprocessing</sub><br>
    <sub>✅ Modeling </sub><br>
    <sub>✅ Organize PPT </sub><br>
</td>
<td align="center">
    <sub><b>📌 담당 역할</b></sub><br>
    <sub>✅ EDA </sub><br>
    <sub>✅ Preprocessing</sub><br>
    <sub>✅ Modeling </sub><br>
    <sub>✅ Organize PPT </sub><br>
</td>
<td align="center">
    <sub><b>📌 담당 역할</b></sub><br>
    <sub>✅ Organize PPT </sub><br>
</td>
<td align="center">
    <sub>✅ Organize PPT </sub><br>
</td>
</tr>
</table>

</div>


## 0\. Overview

### Environment

<p align="left"> <img src="https://img.shields.io/badge/python-3.8-blue?logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/pytorch-2.1.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/> <img src="https://img.shields.io/badge/Transformers-4.35-FFD21E?logo=huggingface&logoColor=white" alt="Transformers"/> <img src="https://img.shields.io/badge/Pandas-1.5-150458?logo=pandas&logoColor=white" alt="Pandas"/> <img src="https://img.shields.io/badge/W&B-20232A?logo=wandb&logoColor=white" alt="Weights & Biases"/> </p>

### Communication Tools
<p align="left"> <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack"/> <img src="https://img.shields.io/badge/Zoom-2D8CFF?style=for-the-badge&logo=zoom&logoColor=white" alt="Zoom"/> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white" alt="Notion"/> </p>

## 1\. Competition Info

### 📌 Overview

**Dialogue Summarization** 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다.

일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

이번 대회를 통해 우리는 **일상 대화를 바탕으로 요약문을 생성하는 모델**을 구축합니다.

### 🎯 Goal

  * **Input:** 2명 이상의 화자가 참여한 Multi-turn 일상 대화 텍스트
  * **Output:** 대화의 핵심 맥락을 포함한 요약문 (Abstractive Summarization)
  * **Objective:** 비정형 텍스트 데이터의 특성을 고려하여 일반화된 요약 성능을 가진 모델 개발

### 📏 Evaluation Metric

모델의 성능 평가는 **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** 점수를 사용합니다.
본 대회에서는 아래 3가지 Metric의 평균 점수를 최종 점수로 산출합니다.

1.  **ROUGE-1 F1:** Unigram(단어) 단위의 중복도
2.  **ROUGE-2 F1:** Bigram(두 단어) 단위의 중복도
3.  **ROUGE-L F1:** LCS(Longest Common Subsequence) 기법을 이용한 최장 길이 매칭

## 2\. Directory Structure

```
├── data/
├── v9_mbart_solar_data/
│   ├── scripts/
│   │   ├── # 1. Configuration Files (실험 설정)
│   │   ├── v9_config_pretrain_en.yaml  # Phase 1: 영어(SAMSum) 프리트레인 설정
│   │   ├── v9_config_aihub.yaml        # Phase 2: 한국어(AI Hub) 도메인 적응 설정
│   │   ├── v9_config_finetune_ko.yaml  # Phase 3: 최종 파인튜닝 설정 (Main)
│   │   ├── v9_config_r3f.yaml          # R3F(Robust Fine-tuning) 실험 설정
│   │   │
│   │   ├── # 2. Training Scripts (학습)
│   │   ├── v9_train.py                 # 메인 학습 스크립트 (Single Model)
│   │   ├── v9_train_kfold.py           # K-Fold 교차 검증 학습 스크립트
│   │   ├── train_cleaned.py            # 정제된 데이터 기반 학습 스크립트
│   │   ├── v9_train_r3f.py             # R3F 적용 학습 스크립트 (Experimental)
│   │   │
│   │   ├── # 3. Inference Scripts (추론)
│   │   ├── v9_inference.py             # 기본 추론 스크립트
│   │   ├── v9_inference_kfold.py       # K-Fold 모델 앙상블 추론
│   │   ├── v9_inference_tuning.py      # 하이퍼파라미터 튜닝용 추론
│   │   ├── v9_inference_fixtag.py      # 화자 태그(#Person1#) 오류 수정 후처리 포함 추론
│   │   ├── v9_inference_foldfix.py     # 특정 Fold 결과 보정 추론
│   │   │
│   │   ├── # 4. Ensemble & Post-processing (앙상블 및 후처리)
│   │   ├── ensemble.py                 # 기본 앙상블 스크립트
│   │   ├── v9_ensemble.py              # v9 모델 전용 앙상블
│   │   ├── high_ensemble.py            # 고득점 모델 가중치 기반 앙상블 (Best Score)
│   │   ├── fix_korean_spacing.py       # 한국어 띄어쓰기 및 구두점 교정 스크립트
│   │   └── check_train.ipynb           # 학습 로그 및 데이터 분포 확인 노트북
│   │
│   └── solar_augumentation/            # Solar API 활용 데이터 증강
└── README.md
```

## 3\. Data Description

### 📊 Dataset Overview & EDA

대화 요약 데이터셋을 분석한 결과, \*\*길이(Length)\*\*와 **주제(Topic)** 측면에서 심각한 불균형을 확인했습니다.

#### 1\. Length Distribution (길이 분포)-AI-부트캠프-16기NLP경진대회-발표.pptx.pdf]

  * **Dialogue (대화문):** 평균 **406자**. 대부분 200\~600자 사이에 분포하지만, 최대 **2,000자 이상**인 Long-tail 데이터가 존재하여 긴 문맥 처리가 중요함.
  * **Summary (요약문):** 평균 **86자**. 대부분 50\~150자로 짧고 매우 정제된 형태를 띰.
  * **Insight:** "길게 읽고(Encoder) 짧게 압축하는(Decoder)" 능력과, 1024 토큰 이상의 긴 시퀀스 처리가 성능의 핵심.

#### 2\. Topic Imbalance (주제 불균형)-AI-부트캠프-16기NLP경진대회-발표.pptx.pdf]

  * **Train vs Dev:** Train과 Dev 데이터셋 간의 \*\*Topic 중복률이 31%\*\*에 불과함 (149개만 겹침).
  * **Minor Topics:** 특정 주제(음식 주문 등)에 편향되어 있으며, 데이터가 5개 미만인 희귀 주제가 다수 존재.
  * **Insight:** 단순히 주제를 암기하는 방식은 Dev 셋에서 성능이 하락함. 주제 정보(Topic)를 모델 입력에서 제외하고, 일반화 능력을 키우는 전략 채택.

## 4\. Preprocessing Strategy

### 🧹 Text Cleaning & Normalization

한국어 구어체 특유의 노이즈를 제거하고 모델이 의미를 파악하기 쉽도록 정규화를 수행했습니다.

  * **감정 표현 정규화:** `ㅎㅎ`, `ㅋㅋ` 등의 자음 남발을 `웃기다`와 같은 의미 있는 단어로 치환-AI-부트캠프-16기NLP경진대회-발표.pptx.pdf].
  * **특수문자 및 공백 처리:** 불필요한 반복 문자(`!!!`, `...`) 축소 및 이중 공백 제거 (`fix_korean_spacing.py`).
  * **Special Tokens:** `#Person1#`, `#PhoneNumber#` 등 화자 및 주요 엔티티 태그를 보존하여 모델이 핵심 정보를 놓치지 않도록 처리.

### 🔢 Tokenizer Efficiency

  * **KoBART vs mBART:** KoBART는 영어 이름(예: `Christine`)을 `['C', 'h', 'r', '...']`로 과도하게 분절(5 tokens)하는 반면, mBART는 `['Christine']`(1 token)으로 효율적으로 처리함-AI-부트캠프-16기NLP경진대회-발표.pptx.pdf].
  * **Decision:** 번역체 데이터 특성상 영어/한국어가 혼용되므로 **mBART-50 Tokenizer** 채택.

## 5\. Modeling

### 🏗️ Model Architecture

  * **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`
  * **Objective:** Conditional Generation (Seq2Seq)

### ⚙️ Hyperparameters (v9 Final)

최종 성능을 달성한 `v9_config_finetune_ko.yaml` 설정값입니다.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Max Input Length** | `1024` | 긴 대화 문맥을 최대한 보존 |
| **Max Target Length** | `128` | 요약문의 평균 길이(86자)를 고려한 설정 |
| **Epochs** | `5` | 과적합 방지를 위한 조기 종료 고려 |
| **Batch Size** | `4` | GPU 메모리 효율성 고려 |
| **Gradient Accumulation** | `8` | Effective Batch Size = 32로 안정적 학습 유도 |
| **Learning Rate** | `3e-5` | Pre-trained 가중치 보존을 위한 낮은 학습률 |
| **Weight Decay** | `0.01` | Regularization |
| **Label Smoothing** | `0.1` | 일반화 성능 향상 (Overconfidence 방지) |

### 🚀 Curriculum Learning (3-Stage Training)

단순 Fine-tuning이 아닌, 모델이 요약의 본질부터 단계적으로 학습하도록 설계했습니다.

1.  **Phase 1 (English Transfer):** `SAMSum` (영어 대화 요약) 데이터로 대화 요약의 논리 학습
2.  **Phase 2 (Korean Adaptation):** `AI Hub` 대규모 한국어 데이터로 한국어 문체 적응
3.  **Phase 3 (Task Fine-tuning):** 대회 데이터 + **Solar 증강 데이터**로 최종 튜닝

## 6\. Result

### 🏆 Final Leaderboard

최종적으로 **mBART 기반 Curriculum Learning 모델**과 **Recursive Ensemble** 전략을 통해 리더보드 2위를 달성했습니다.

| Rank | Team | ROUGE-1 | ROUGE-2 | ROUGE-L | Final Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **🥈 2** | **VIBE** | **0.5935** | **0.4064** | **0.5164** | **47.7938** |

### Presentation
[![Presentation](https://github.com/user-attachments/assets/d8ed4d2d-be00-413b-b0ec-5c1c23f9dc33)]([https://docs.google.com/presentation/d/1Pc4JzGhqG0PY4wVeC9hVAeT_XcljLxX6/edit?slide=id.g37012e7854d_7_60#slide=id.g37012e7854d_7_60](https://docs.google.com/presentation/d/1_fcTPc6iv5WsT8ROrzDJOtV-eNysgY8a/edit))
> *위 버튼을 클릭하면 상세한 발표 자료(PDF)를 볼 수 있습니다.*

## etc

### Reference

  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
  - [mBART Paper](https://arxiv.org/abs/2001.08210)
  - [R-Drop Paper](https://arxiv.org/abs/2106.14448)
  - [Solar API](https://console.upstage.ai/)
