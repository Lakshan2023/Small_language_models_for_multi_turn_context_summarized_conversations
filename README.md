# Small Language Models for Multi-turn Context-Summarized Customer Service QA

This repository contains the code and evaluation framework for research on evaluating Small Language Models for context-summarized multi-turn customer service question answering.

## Overview

This study evaluates instruction-tuned Small Language Models (SLMs) for context-summarized multi-turn customer service question answering. We compare 9 fine-tuned SLMs against 3 commercial LLMs using comprehensive evaluation methods including lexical metrics, semantic similarity, LLM-as-a-judge, and human evaluation.

## Key Features

- **Synthetic Dataset Pipeline**: Transforms single-turn QA into multi-turn conversations with context summarization
- **Comprehensive Evaluation**: Combines automatic metrics with qualitative assessments (lexical, semantic, LLM-as-a-judge, human evaluation)
- **Stage-Based Evaluation Framework**: Novel conversation stage segmentation (Early/Mid/Late) to analyze model behavior across different phases of customer service interactions
- **Fine-Tuning Framework**: QLoRA-based parameter-efficient fine-tuning for SLMs

## Models Evaluated

### Small Language Models (SLMs)
- LLaMA-3.2-1B, 3B-Instruct and LLaMA-3.1-8B-Instruct
- Qwen-3-1.7B, 4B, and 8B-Instruct
- Phi-4-Mini (3.8B)
- Gemma-3-4B-Instruct
- SmolLM3-3B-Instruct

### Commercial LLMs (Baseline)
- GPT-4.1
- Gemini-2.5-Flash
- Virtuoso-Large

## Repository Structure

```
├── ContextSummarizationAblationStudy/
│   ├── ContextSummarizationModelEvaluation/
│   │   ├── Llama-3.1_8B_evaluation.ipynb
│   │   ├── Llama-3.2_3B_evaluation.ipynb
│   │   ├── Phi-4-mini_evaluation.ipynb
│   │   ├── Qwen-3-4B_evaluation.ipynb
│   │   ├── Qwen-3-8B_evaluation.ipynb
│   │   ├── context_summarization_evaluation_results.ipynb
│   │   ├── context_summary_llm_judge.ipynb
│   │   ├── gemini_evaluation.ipynb
│   │   └── gpt4.1_evaluation.ipynb
│   ├── ContextSummarizationModelTraining/
│   │   ├── Llama-3.1-8B-Instruct-model.ipynb
│   │   ├── Phi-4-mini-instruct.ipynb
│   │   ├── Qwen-3-4B-model-Instruct.ipynb
│   │   ├── Qwen-3-8B-Instruct-model.ipynb
│   │   └── llama-3.2-3B-Instruct-model.ipynb
│   └── ContextSummarizationDataset.ipynb
│
├── DatasetCreation/
│   ├── CustomerSupportDataset.ipynb
│   └── context-summary-generation.ipynb
│
├── InferenceCostBenchmark/
│   ├── Llama32_1b_benchmark.ipynb
│   ├── Llama_31_8b_instruct_benchmark.ipynb
│   ├── Llama_32_3b_instruct_benchmark.ipynb
│   ├── create_test_dataset.ipynb
│   ├── gemma_3_4b_instruct_benchmark.ipynb
│   ├── phi_4_mini_benchmark.ipynb
│   ├── qwen3_17b_instruct_benchmark.ipynb
│   ├── qwen3_4b_instruct_benchmark.ipynb
│   ├── qwen3_8b_instruct_benchmark.ipynb
│   └── smollm3_3b_instruct_benchmark.ipynb
│
├── ModelDecodingEvaluation/
│   ├── Gemma3-4B-instruct_evaluation.ipynb
│   ├── Llama-3.1_8B_evaluation.ipynb
│   ├── Llama-3.2_1B_evaluation.ipynb
│   ├── Llama-3.2_3B_evaluation.ipynb
│   ├── Phi-4-mini_evaluation.ipynb
│   ├── Qwen-3-1.7B_evaluation.ipynb
│   ├── Qwen-3-4B_evaluation.ipynb
│   ├── Qwen-3-8B_evaluation.ipynb
│   ├── SmolLM3-3B_evaluation.ipynb
│   ├── gemini_evaluation.ipynb
│   ├── gpt4.1_evaluation.ipynb
│   └── virtuoso_large_evaluation.ipynb
│
├── OverallEvaluationResults/
│   ├── EvaluationPlots.ipynb
│   ├── HumanEvaluation.ipynb
│   ├── PairwiseEvaluation.ipynb
│   └── judge_semantic_lexical_results.ipynb
│
├── SLMsFinetuning/
│   ├── Gemma3-4B-instruct-model.ipynb
│   ├── Llama-3.1-8B-Instruct-model.ipynb
│   ├── Phi-4-mini-instruct.ipynb
│   ├── Qwen-3-1.7B-model-Instruct.ipynb
│   ├── Qwen-3-4B-model-Instruct.ipynb
│   ├── Qwen-3-8B-Instruct-model.ipynb
│   ├── SmolLM3-3B-Instruct.ipynb
│   ├── llama-3.2-1B-Instruct-model.ipynb
│   └── llama-3.2-3B-Instruct-model.ipynb
│
├── LICENSE
└── README.md
```

## Dataset

The synthetic dataset is constructed from the Hugging Face TalkMap Customer Service Banking Conversation Corpus.

![Synthetic Context-Summarized Multi-turn QA pipeline](figures/datapipeline.png)

**Construction Pipeline:**
1. **Preprocessing and Filtering:** Retained conversations ranging from 5 to 100 turns to ensure realistic dialogue depth and applied Regex-based noise removal.
2. **Multi-Turn Construction:** Aggregated sequential single turns into complete dialogues, applied de-duplication, and partitioned conversations into early (20%), middle (70%), and late (10%) segments.
3. **Context Summarization:** Summarized prior conversational histories using GPT-4o-mini to condense token length while preserving essential facts, names, and verification steps.
4. **Response Refinement & Moderation:** Enhanced agent answers for naturalness, clarity, and contextual coherence using GPT-4.1, followed by safety filtering using OpenAI’s Moderation API.
5. **Structured Instance Formation:** Assembled standard QA instances (instruction, summarized history, current query, refined response) and divided them into standard splits.

 The created dataset is publicly available at: [Lakshan2003/customer-support-client-agent-conversations](https://huggingface.co/datasets/Lakshan2003/customer-support-client-agent-conversations).
 
**Dataset Statistics:**
- Training: 128,335 samples
- Validation: 18,333 samples  
- Test: 36,669 samples
- Average turns per conversation: ~10

 ## Model Training & Inference

 ![QLoRA training pipeline](figures/qlorapipeline.png)

**Parameter-Efficient Fine-Tuning (QLoRA):**
All Small Language Models (SLMs) were adapted using Quantized Low-Rank Adaptation (QLoRA) to enable efficient domain adaptation on constrained hardware.
- **Quantization & Adapters:** 4-bit precision base models with LoRA adapters injected into attention and feed-forward layers (Rank = 16, Alpha = 32, Dropout = 0.1).
- **Training Hyperparameters:** AdamW 8-bit optimizer, learning rate of 2×10⁻⁵ (cosine scheduler), 3 epochs, effective batch size of 16, and a max sequence length of 512 tokens.
- **Framework & Hardware:** Training was conducted using Unsloth and Hugging Face frameworks on a single NVIDIA RTX A100 40GB GPU.

**Inference Configuration:**
Inference was conducted on the full test split (36,669 instances) using a maximum generation length of 128 tokens across all models to ensure concise, customer-service-appropriate responses.
- **SLM Decoding:** Configured based on publisher recommendations for stability. 
- **Commercial LLM Decoding:** API-based inference for GPT-4.1, Gemini-2.5-Flash, and Virtuoso-Large standardized at Temp 0.7 and Top-p 0.9. 
- **Reasoning Constraint:** Gemini-2.5-Flash's explicit reasoning behavior was disabled (thinking budget = 0) to align with the direct instruction-following setup of the SLMs.

## Evaluation Metrics

### Automatic Metrics
- **Lexical**: ROUGE-L, METEOR
- **Semantic**: BERTScore, BARTScore, Cosine Similarity

### Qualitative Assessment
- **LLM-as-a-Judge**: Using Claude Sonnet 4.5
- **Human Evaluation**: 3 independent evaluators
- **Pairwise Comparison**: Using Claude Haiku 4.5

### Evaluation Dimensions
1. Human-Likeness
2. Continuity and Context Understanding
3. Tone and Clarity
4. Task Appropriateness

## Stage-Based Evaluation

A key contribution of this work is the **conversation stage-based evaluation framework** that segments interactions into three distinct phases:

### Conversation Stages
- **Early Stage (10%)**: Issue identification and initial context gathering
- **Mid Stage (80%)**: Core interaction with substantive information exchange — most challenging phase requiring strongest contextual reasoning
- **Late Stage (10%)**: Resolution and closure

### Stage-Based Insights
- **Early Stage**: Top SLMs show moderate competitiveness in issue identification (LLM-as-a-Judge scores: 3.7–3.8)
- **Mid Stage**: Most challenging phase — largest gaps observed in Continuity, Context Understanding and Task Appropriateness; LLaMA-3.1-8B and Qwen-3-8B maintain the strongest SLM performance
- **Late Stage**: SLMs demonstrate their strongest results; LLaMA-3.2-3B-Instruct, Phi-4-Mini and Qwen-3-4B-Instruct exceed Gemini-2.5-Flash under human evaluation (scores above 4.5)

This segmentation enables targeted analysis beyond overall performance scores, identifying which models excel at specific conversation phases.

## Key Results (Summary)

### Quantitative Evaluation (Full Test Set — 36,669 instances)

| Model | ROUGE-L (↑) | METEOR (↑) | BARTScore (↑) | BERTScore F1 (↑) | Cosine Sim. (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaMA-3.2-1B-Instruct | 0.2332 | 0.3032 | -2.7060 | 0.8821 | 0.5909 |
| Qwen-3-1.7B-Instruct | 0.3697 | 0.4138 | -2.3096 | 0.9096 | 0.6731 |
| LLaMA-3.2-3B-Instruct | 0.3842 | 0.4471 | -2.2655 | 0.9121 | 0.6958 |
| SmolLM3-3B | 0.2393 | 0.3022 | -2.7699 | 0.8830 | 0.5428 |
| Phi-4-Mini (3.8B) | 0.3747 | 0.4303 | -2.2872 | 0.9107 | 0.6891 |
| Qwen-3-4B-Instruct | **0.3959** | 0.4455 | **-2.2311** | **0.9137** | 0.6972 |
| Gemma-3-4B-Instruct | 0.2024 | 0.2782 | -3.0766 | 0.8752 | 0.5134 |
| LLaMA-3.1-8B-Instruct | 0.3940 | **0.4569** | -2.2332 | 0.9134 | **0.7051** |
| Qwen-3-8B-Instruct | 0.3121 | 0.3792 | -2.4970 | 0.8995 | 0.6621 |
| GPT-4.1 | 0.3038 | 0.3685 | -2.5145 | 0.8994 | 0.6749 |
| Gemini-2.5-Flash | 0.2771 | 0.3110 | -2.6409 | 0.8942 | 0.6234 |
| Virtuoso-Large | 0.3161 | 0.3770 | -2.4625 | 0.9011 | 0.6676 |

*Comparison of lexical and semantic similarity results on the complete test set. Models are grouped by size: small models (<4B), 8B models, and commercial LLMs.*

Fine-tuned SLMs consistently outperform commercial LLMs on quantitative metrics due to domain-specific fine-tuning alignment with reference responses.

---

### LLM-as-a-Judge Evaluation (Claude Sonnet 4.5 — 6,000 samples per model, 1–5 Likert scale)

| Model | Human Likeness | Continuity & Context Understanding | Tone & Clarity | Task Appropriateness | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaMA-3.2-1B-Instruct | 3.165 | 2.358 | 3.342 | 2.171 | 2.759 |
| Qwen-3-1.7B-Instruct | 3.738 | 3.362 | 3.818 | 2.994 | 3.478 |
| SmolLM3-3B | 2.654 | 1.772 | 2.717 | 1.696 | 2.210 |
| LLaMA-3.2-3B-Instruct | 4.075 | 3.480 | 4.105 | 3.212 | 3.718 |
| Phi-4-Mini (3.8B) | 3.988 | 3.360 | 4.034 | 3.093 | 3.619 |
| Qwen-3-4B-Instruct | 4.044 | 3.430 | 4.071 | 3.170 | 3.679 |
| Gemma-3-4B-Instruct | 2.582 | 1.729 | 2.597 | 1.673 | 2.145 |
| LLaMA-3.1-8B-Instruct | 4.115 | 3.591 | 4.149 | 3.322 | 3.794 |
| Qwen-3-8B-Instruct | 3.950 | 3.648 | 4.067 | 3.306 | 3.743 |
| GPT-4.1 | **4.316** | **4.079** | **4.381** | **3.808** | **4.146** |
| Gemini-2.5-Flash | 4.054 | 3.742 | 4.101 | 3.180 | 3.769 |
| Virtuoso-Large | 4.171 | 3.864 | 4.204 | 3.530 | 3.942 |

*Overall LLM-as-a-judge evaluation results across four qualitative dimensions using a 5-point Likert scale. Models are grouped by size: small models (<4B), 8B models, and commercial LLMs.*

---

### Human Evaluation (3 Independent Evaluators — 500 samples per model, 1–5 Likert scale)

| Model | Human Likeness | Continuity & Context Understanding | Tone & Clarity | Task Appropriateness | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SmolLM3-3B | 3.003 | 2.615 | 2.965 | 2.261 | 2.711 |
| LLaMA-3.2-3B-Instruct | 4.250 | 4.325 | 4.286 | 3.721 | 4.146 |
| Phi-4-Mini (3.8B) | 4.164 | 4.303 | 4.215 | 3.553 | 4.059 |
| Qwen-3-4B-Instruct | 4.203 | 4.264 | 4.230 | 3.579 | 4.069 |
| Gemma-3-4B-Instruct | 3.110 | 2.520 | 2.968 | 2.146 | 2.686 |
| GPT-4.1 | **4.674** | **4.827** | **4.722** | **4.286** | **4.627** |
| Gemini-2.5-Flash | 4.181 | 4.567 | 4.247 | 3.770 | 4.191 |
| Virtuoso-Large | 4.507 | 4.726 | 4.637 | 4.249 | 4.529 |

*Overall Human evaluation results across four qualitative dimensions using a 5-point Likert scale.

---

### Pairwise Evaluation (Claude Haiku 4.5 — 1,000 samples)

Selected highlights (SLM win % against commercial LLMs):

| LLM | SLM | LLM Wins (%) | SLM Wins (%) | Ties (%) |
| :--- | :--- | :---: | :---: | :---: |
| Gemini-2.5-Flash | LLaMA-3.2-1B-Instruct | 43.10 | 38.60 | 18.30 |
| Gemini-2.5-Flash | Qwen-3-1.7B-Instruct | 49.80 | 33.50 | 16.70 |
| Gemini-2.5-Flash | LLaMA-3.2-3B-Instruct | 37.40 | 49.70 | 12.90 |
| Gemini-2.5-Flash | Phi-4-Mini (3.8B) | 40.50 | 43.90 | 15.60 |
| Gemini-2.5-Flash | Qwen-3-4B-Instruct | 41.20 | 45.00 | 13.80 |
| Gemini-2.5-Flash | LLaMA-3.1-8B-Instruct | 28.60 | 52.90 | 18.50 |
| Gemini-2.5-Flash | Qwen-3-8B-Instruct | 26.60 | 55.80 | 17.60 |
| GPT-4.1 | LLaMA-3.2-1B-Instruct | 68.60 | 15.70 | 15.70 |
| GPT-4.1 | Qwen-3-1.7B-Instruct | 79.00 | 8.70 | 12.30 |
| GPT-4.1 | LLaMA-3.2-3B-Instruct | 67.00 | 17.90 | 15.10 |
| GPT-4.1 | Phi-4-Mini (3.8B) | 72.50 | 14.90 | 12.60 |
| GPT-4.1 | Qwen-3-4B-Instruct | 70.90 | 14.60 | 14.50 |
| GPT-4.1 | LLaMA-3.1-8B-Instruct | 61.30 | 19.00 | 19.70 |
| GPT-4.1 | Qwen-3-8B-Instruct | 54.20 | 23.80 | 22.00 |
| Virtuoso-Large | LLaMA-3.2-1B-Instruct | 61.60 | 19.80 | 18.60 |
| Virtuoso-Large | Qwen-3-1.7B-Instruct | 73.40 | 12.80 | 13.80 |
| Virtuoso-Large | LLaMA-3.2-3B-Instruct | 56.90 | 24.60 | 18.50 |
| Virtuoso-Large | Phi-4-Mini (3.8B) | 64.70 | 19.90 | 15.40 |
| Virtuoso-Large | Qwen-3-4B-Instruct | 62.60 | 21.80 | 15.60 |
| Virtuoso-Large | LLaMA-3.1-8B-Instruct | 54.70 | 28.00 | 17.30 |
| Virtuoso-Large | Qwen-3-8B-Instruct | 46.70 | 31.90 | 21.40 |

*Pairwise LLM vs. SLM evaluation results expressed as win percentages.*

LLaMA-3.1-8B and Qwen-3-8B both outperform Gemini-2.5-Flash in direct pairwise comparisons. No SLM exceeds GPT-4.1 or Virtuoso-Large in win rate.

---

### Stage-wise Performance Summary (LLM-as-a-Judge Overall Mean)

| Stage | Model | Human Likeness | Continuity & Context Understanding | Tone & Clarity | Task Appropriateness | Overall Mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Early-stage** | LLaMA-3.2-1B-Instruct | 3.625 | 2.927 | 3.725 | 2.583 | 3.215 |
| | Qwen-3-1.7B-Instruct | 3.880 | 3.508 | 3.957 | 3.145 | 3.622 |
| | LLaMA-3.2-3B-Instruct | 4.125 | 3.625 | 4.160 | 3.288 | 3.800 |
| | SmolLM3-3B | 2.227 | 1.578 | 2.290 | 1.503 | 1.900 |
| | Qwen-3-4B-Instruct | 4.100 | 3.575 | 4.130 | 3.253 | 3.764 |
| | Phi-4-Mini | 4.058 | 3.473 | 4.113 | 3.158 | 3.700 |
| | Gemma-3-4B-Instruct | 2.517 | 1.553 | 2.557 | 1.508 | 2.034 |
| | LLaMA-3.1-8B-Instruct | 4.152 | 3.613 | 4.202 | 3.310 | 3.819 |
| | Qwen-3-8B-Instruct | 3.932 | 3.513 | 4.038 | 3.175 | 3.665 |
| | GPT-4.1 | **4.310** | **4.018** | **4.383** | **3.715** | **4.106** |
| | Virtuoso-Large | 4.157 | 3.750 | 4.185 | 3.425 | 3.879 |
| | Gemini-2.5-Flash | 4.143 | 3.768 | 4.185 | 3.350 | 3.862 |
| **Mid-Stage** | LLaMA-3.2-1B-Instruct | 3.139 | 2.327 | 3.315 | 2.150 | 2.733 |
| | Qwen-3-1.7B-Instruct | 3.684 | 3.277 | 3.766 | 2.900 | 3.407 |
| | LLaMA-3.2-3B-Instruct | 4.034 | 3.388 | 4.065 | 3.104 | 3.648 |
| | SmolLM3-3B | 2.738 | 1.806 | 2.798 | 1.719 | 2.265 |
| | Qwen-3-4B-Instruct | 4.001 | 3.342 | 4.030 | 3.061 | 3.609 |
| | Phi-4-Mini | 3.943 | 3.267 | 3.991 | 2.980 | 3.545 |
| | Gemma-3-4B-Instruct | 2.618 | 1.737 | 2.624 | 1.659 | 2.160 |
| | LLaMA-3.1-8B-Instruct | 4.077 | 3.525 | 4.111 | 3.236 | 3.737 |
| | Qwen-3-8B-Instruct | 3.924 | 3.625 | 4.052 | 3.249 | 3.712 |
| | GPT-4.1 | **4.286** | **4.056** | **4.355** | **3.764** | **4.115** |
| | Virtuoso-Large | 4.137 | 3.835 | 4.173 | 3.470 | 3.904 |
| | Gemini-2.5-Flash | 4.039 | 3.759 | 4.081 | 3.127 | 3.752 |
| **Late-stage** | LLaMA-3.2-1B-Instruct | 2.910 | 2.033 | 3.173 | 1.930 | 2.512 |
| | Qwen-3-1.7B-Instruct | 4.032 | 3.902 | 4.095 | 3.591 | 3.905 |
| | LLaMA-3.2-3B-Instruct | 4.357 | 4.068 | 4.373 | 4.002 | 4.200 |
| | SmolLM3-3B | 2.410 | 1.688 | 2.500 | 1.700 | 2.074 |
| | Qwen-3-4B-Instruct | 4.335 | 3.993 | 4.340 | 3.950 | 4.154 |
| | Phi-4-Mini | 4.273 | 3.983 | 4.297 | 3.932 | 4.121 |
| | Gemma-3-4B-Instruct | 2.363 | 1.843 | 2.418 | 1.945 | 2.142 |
| | LLaMA-3.1-8B-Instruct | 4.386 | 4.102 | 4.397 | 4.030 | 4.229 |
| | Qwen-3-8B-Instruct | 4.182 | 3.967 | 4.212 | 3.885 | 4.062 |
| | GPT-4.1 | **4.563** | **4.323** | **4.588** | **4.253** | **4.432** |
| | Virtuoso-Large | 4.453 | 4.212 | 4.468 | 4.115 | 4.312 |
| | Gemini-2.5-Flash | 4.085 | 3.577 | 4.177 | 3.433 | 3.818 |

*Stage-wise LLM-as-a-judge evaluation results across early, mid, and late-stage customer-service interactions using a 5-point Likert scale. Scores are averaged over 6,000 evaluation samples, with 600 early-stage, 4,800 mid-stage, and 600 late-stage instances.*

SLMs perform **weakest in Mid-stage** and **strongest in Late-stage** interactions. LLaMA-3.1-8B-Instruct surpasses Gemini-2.5-Flash in Late-stage (4.229 vs 3.818).

---
### Conversation Stage-based Human Evaluation Results

| Stage | Model | Human Likeness | Continuity & Context | Tone & Clarity | Task Appropriateness | Overall Mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Early-Stage** | LLaMA-3.2-3B-Instruct | 4.107 | 4.227 | 4.093 | 3.593 | 4.005 |
| | SmolLM3-3B | 2.513 | 2.173 | 2.487 | 1.833 | 2.252 |
| | Qwen-3-4B-Instruct | 4.073 | 4.060 | 4.093 | 3.433 | 3.915 |
| | Phi-4-Mini | 3.940 | 4.147 | 4.000 | 3.440 | 3.882 |
| | Gemma-3-4B-Instruct | 3.013 | 2.360 | 2.887 | 1.953 | 2.553 |
| | GPT-4.1 | **4.467** | **4.753** | **4.540** | **4.073** | **4.458** |
| | Virtuoso-Large | 4.353 | 4.680 | 4.507 | 4.100 | 4.410 |
| | Gemini-2.5-Flash | 4.187 | 4.527 | 4.240 | 3.933 | 4.222 |
| **Mid-stage** | LLaMA-3.2-3B-Instruct | 4.217 | 4.291 | 4.258 | 3.650 | 4.104 |
| | SmolLM3-3B | 3.098 | 2.694 | 3.059 | 2.323 | 2.793 |
| | Qwen-3-4B-Instruct | 4.164 | 4.249 | 4.197 | 3.510 | 4.030 |
| | Phi-4-Mini | 4.138 | 4.274 | 4.187 | 3.478 | 4.019 |
| | Gemma-3-4B-Instruct | 3.141 | 2.531 | 3.002 | 2.143 | 2.704 |
| | GPT-4.1 | **4.675** | **4.846** | **4.725** | **4.283** | **4.632** |
| | Virtuoso-Large | 4.489 | 4.723 | 4.631 | 4.223 | 4.517 |
| | Gemini-2.5-Flash | 4.159 | 4.583 | 4.233 | 3.731 | 4.177 |
| **Late-stage** | LLaMA-3.2-3B-Instruct | 4.660 | 4.700 | 4.700 | 4.420 | 4.620 |
| | SmolLM3-3B | 2.733 | 2.427 | 2.693 | 2.193 | 2.512 |
| | Qwen-3-4B-Instruct | 4.640 | 4.587 | 4.627 | 4.273 | 4.532 |
| | Phi-4-Mini | 4.593 | 4.687 | 4.653 | 4.260 | 4.548 |
| | Gemma-3-4B-Instruct | 2.960 | 2.593 | 2.780 | 2.360 | 2.673 |
| | GPT-4.1 | **4.867** | **4.753** | **4.880** | **4.520** | **4.755** |
| | Virtuoso-Large | 4.800 | 4.793 | 4.820 | 4.600 | 4.753 |
| | Gemini-2.5-Flash | 4.347 | 4.480 | 4.367 | 3.920 | 4.278 |

*Stage-wise human evaluation results across early, mid, and late-stage customer-service interactions using a 5-point Likert scale. Scores are averaged over 500 evaluation samples per model, consisting of 50 early-stage, 400 mid-stage, and 50 late-stage instances.*

---

### Conversation Stage-based Pairwise Evaluation Results

| Tested SLM | Early (LLM%) | Early (SLM%) | Early (Tie%) | Mid (LLM%) | Mid (SLM%) | Mid (Tie%) | Late (LLM%) | Late (SLM%) | Late (Tie%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vs. GPT-4.1** | | | | | | | | | |
| LLaMA-3.2-1B-Instruct | 65.66 | 17.17 | 17.17 | 72.01 | 12.94 | 15.05 | 43.30 | 37.11 | 19.59 |
| Qwen-3-1.7B-Instruct | 75.76 | 11.11 | 13.13 | 82.34 | 6.84 | 10.82 | 54.64 | 21.65 | 23.71 |
| LLaMA-3.2-3B-Instruct | 72.73 | 18.18 | 9.09 | 67.91 | 16.54 | 15.55 | 53.61 | 28.87 | 17.53 |
| Phi-4-Mini (3.8B) | 71.72 | 13.13 | 15.15 | 74.75 | 13.68 | 11.57 | 54.64 | 26.80 | 18.56 |
| Qwen-3-4B-Instruct | 66.67 | 14.14 | 19.19 | 72.89 | 13.43 | 13.68 | 58.76 | 24.74 | 16.49 |
| LLaMA-3.1-8B-Instruct | 54.55 | 22.22 | 23.23 | 64.05 | 18.16 | 17.79 | 45.36 | 22.68 | 31.96 |
| Qwen-3-8B-Instruct | 54.55 | 19.19 | 26.26 | 54.35 | 24.25 | 21.39 | 52.58 | 24.74 | 22.68 |
| **vs. Virtuoso-Large** | | | | | | | | | |
| LLaMA-3.2-1B-Instruct | 57.58 | 25.25 | 17.17 | 64.43 | 17.29 | 18.28 | 42.27 | 35.05 | 22.68 |
| Qwen-3-1.7B-Instruct | 65.66 | 14.14 | 20.20 | 76.37 | 12.81 | 10.82 | 56.70 | 11.34 | 31.96 |
| LLaMA-3.2-3B-Instruct | 58.59 | 27.27 | 14.14 | 57.84 | 23.63 | 18.53 | 47.42 | 29.90 | 22.68 |
| Phi-4-Mini (3.8B) | 57.58 | 26.26 | 16.16 | 67.79 | 18.91 | 13.31 | 46.39 | 21.65 | 31.96 |
| Qwen-3-4B-Instruct | 56.57 | 27.27 | 16.16 | 65.17 | 20.52 | 14.30 | 47.42 | 26.80 | 25.77 |
| LLaMA-3.1-8B-Instruct | 47.47 | 34.34 | 18.18 | 57.21 | 26.12 | 16.67 | 41.24 | 37.11 | 21.65 |
| Qwen-3-8B-Instruct | 49.49 | 31.31 | 19.19 | 45.27 | 33.33 | 21.39 | 55.67 | 20.62 | 23.71 |
| **vs. Gemini-2.5-Flash** | | | | | | | | | |
| LLaMA-3.2-1B-Instruct | 49.49 | 35.35 | 15.15 | 43.91 | 38.68 | 17.41 | 29.90 | 41.24 | 28.87 |
| Qwen-3-1.7B-Instruct | 56.57 | 28.28 | 15.15 | 49.50 | 34.08 | 16.42 | 45.36 | 34.02 | 20.62 |
| LLaMA-3.2-3B-Instruct | 42.42 | 44.44 | 13.13 | 36.82 | 50.62 | 12.56 | 37.11 | 47.42 | 15.46 |
| Phi-4-Mini (3.8B) | 45.45 | 37.37 | 17.17 | 40.67 | 43.66 | 15.67 | 34.02 | 52.58 | 13.40 |
| Qwen-3-4B-Instruct | 49.49 | 36.36 | 14.14 | 40.67 | 46.02 | 13.31 | 37.11 | 45.36 | 17.53 |
| LLaMA-3.1-8B-Instruct | 31.31 | 40.40 | 28.28 | 26.99 | 55.97 | 17.04 | 39.18 | 40.21 | 20.62 |
| Qwen-3-8B-Instruct | 44.44 | 38.38 | 17.17 | 23.13 | 60.45 | 16.42 | 37.11 | 35.05 | 27.84 |

*Win and tie percentages for pairwise comparisons between selected high-performing SLMs and commercial LLMs across Early, Mid, and Late conversation stages, using Claude Haiku 4.5 as the judge.*

---

| Model | Avg Latency (s) (↓) | Avg TTFT (s) (↓) | Avg GPU Memory (GB) (↓) | Avg Disk Storage (GB) (↓) | Avg Time per Token (s) (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Llama-3.2-1B-Instruct | **0.94** | **0.02** | **2.36** | **2.35** | **0.02** |
| Qwen3-1.7B-Instruct | 1.83 | 0.05 | 3.90 | 3.87 | 0.04 |
| LLaMA-3.2-3B-Instruct | 1.59 | 0.04 | 6.11 | 6.08 | 0.04 |
| SmolLM3-3B | 2.09 | 0.05 | 5.82 | 5.86 | 0.04 |
| Phi-4-Mini | 2.07 | 0.06 | 7.30 | 7.20 | 0.05 |
| Qwen3-4B-Instruct | 2.32 | 0.06 | 7.70 | 7.63 | 0.06 |
| Gemma-3-4B-Instruct | 4.14 | 0.08 | 24.40 | 8.17 | 0.07 |
| LLaMA-3.1-8B-Instruct | 1.81 | 0.04 | 15.10 | 15.12 | 0.04 |
| Qwen3-8B-Instruct | 2.44 | 0.06 | 15.42 | 15.44 | 0.06 |

*Inference performance metrics across different model sizes. Models are grouped by size: small models (<4B) and 8B models. Lower values indicate better performance for all metrics.*

## Ablation Study: SLM-based Context Summarization

 ![Context summarization QLoRA training pipeline](figures/context_summarization_qlora_pipeline.png)

**Goal:** To determine whether fine-tuned SLMs can effectively replace commercial LLMs in generating accurate and coherent context summaries for multi-turn customer service conversations.

**Models Evaluated:** LLaMA-3.1-8B-Instruct, LLaMA-3.2-3B-Instruct, Phi-4-Mini, Qwen-3-4B-Instruct, and Qwen-3-8B-Instruct (evaluated against Gemini-2.5-Flash and GPT-4.1).

**Dataset:**
- Created from a random selection of 50,000 unique conversations from the main multi-turn corpus.
- **Splits:** 35,000 Training (70%), 5,000 Validation (10%), and 10,000 Test (20%).
- **Link:** [Lakshan2003/customer-support-context-summary-50k](https://huggingface.co/datasets/Lakshan2003/customer-support-context-summary-50k)

**Parameter-Efficient Fine-Tuning (QLoRA):**
- **LoRA Configuration:** Adapters applied to query, key, value, and output projection layers (Rank = 8, Alpha = 16, Dropout = 0.1).
- **Hyperparameters:** Max sequence length of 1,024 tokens, AdamW 8-bit optimizer, learning rate of 2×10⁻⁵ (cosine scheduler, warmup 0.05), 3 epochs, and an effective batch size of 16.

**Evaluation Framework:**
- **Quantitative Evaluation (10,000 test instances):** Lexical and semantic overlap assessed using BLEU, METEOR, ROUGE-L, Cosine Similarity, BERTScore F1, and BARTScore.
- **Qualitative Evaluation (1,000 test subset):** LLM-as-a-judge framework using Claude Sonnet 4.5 on a 1–5 Likert scale across three dimensions:
  1. **Information Accuracy:** Correctly capturing key facts (names, accounts, dates, verification steps).
  2. **Structural Clarity:** Logical and clear organization of the client's issue and current status.
  3. **Faithfulness:** Strict adherence to the original conversation without introducing unsupported assumptions.
 
 ### Quantitative Evaluation: Context Summary Generation

| Model | BLEU (↑) | METEOR (↑) | ROUGE-L (↑) | Cosine Sim. (↑) | BERTScore F1 (↑) | BARTScore (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LLaMA-3.2-3B-Instruct | 0.3663 | 0.5976 | 0.5305 | 0.9255 | 0.9321 | -1.9362 |
| Phi-4-Mini | 0.3620 | 0.5804 | 0.5159 | 0.9147 | 0.9309 | -1.9466 |
| Qwen3-4B-Instruct | 0.3753 | 0.6000 | 0.5332 | 0.9201 | 0.9324 | -1.9236 |
| LLaMA-3.1-8B-Instruct | 0.3791 | 0.6175 | 0.5486 | 0.9264 | 0.9346 | -1.8701 |
| Qwen3-8B-Instruct | **0.4021** | **0.6179** | **0.5609** | **0.9286** | **0.9363** | **-1.8455** |
| Gemini-2.5-Flash | 0.2652 | 0.4730 | 0.4511 | 0.8656 | 0.9194 | -2.1386 |
| GPT-4.1 | 0.2998 | 0.5470 | 0.4706 | 0.9117 | 0.9241 | -2.1292 |

*Lexical and semantic similarity results for context summary generation. Models are grouped by size: small models (<4B), 8B models, and commercial LLMs.*


### Qualitative Evaluation (LLM-as-a-Judge): Context Summary Generation

| Model | Information Accuracy (↑) | Structural Clarity (↑) | Faithfulness (↑) | Overall Score (↑) |
| :--- | :---: | :---: | :---: | :---: |
| LLaMA-3.2-3B-Instruct | 4.5816 | 4.8398 | 4.4825 | 4.6346 |
| Phi-4-Mini | 4.6300 | 4.8760 | 4.5390 | 4.6817 |
| Qwen3-4B-Instruct | 4.6580 | 4.8760 | 4.5640 | 4.6993 |
| LLaMA-3.1-8B-Instruct | 4.6897 | 4.9039 | 4.6296 | 4.7411 |
| Qwen3-8B-Instruct | 4.6927 | 4.8959 | 4.6256 | 4.7381 |
| Gemini-2.5-Flash | **4.7930** | 4.9300 | **4.7700** | 4.8310 |
| GPT-4.1 | 4.8060 | **4.9480** | 4.7510 | **4.8350** |

*LLM-as-a-Judge qualitative evaluation results for multi-turn customer service context summarization using a 1–5 Likert scale. Scores represent the average across 1,000 evaluation instances for each model.*

## Experiments, Models and Datasets

All experiments conducted in this study, including fine-tuned models, inference outputs and constructed datasets, are publicly available at the following repository:

Hugging Face: https://huggingface.co/Lakshan2003


## Citation

If you use this work, please cite:

```bibtex
@misc{cooray2026smalllanguagemodelshandle,
      title={Can Small Language Models Handle Context-Summarized Multi-Turn Customer-Service QA? A Synthetic Data-Driven Comparative Evaluation}, 
      author={Lakshan Cooray and Deshan Sumanathilaka and Pattigadapa Venkatesh Raju},
      year={2026},
      eprint={2602.00665},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.00665}, 
}
```

## License

This project is intended for research purposes.For commercial Usage Please contact the author.

## Acknowledgments

We thank the human evaluators and the Zame AI team for supporting API usage for large-scale model inference and evaluation.
