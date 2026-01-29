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
├── DatasetCreation/
│   ├── CustomerSupportDataset.ipynb    # Dataset construction pipeline
│   └── context-summary-generation.ipynb # Context summarization
│
├── SLMsFintuning/
│   └── [model fine-tuning notebooks]
│
├── ModelDecodingEvaluation/
│   └── [model evaluation notebooks]
│
└── OverallEvaluationResults/
    ├── HumanEvaluation.ipynb
    ├── PairwiseEvaluation.ipynb
    └── judge_semantic_lexical_results.ipynb
```

## Key Results

- Leading 3-8B SLMs achieve near-LLM performance on automatic metrics
- LLaMA-3.2-3B-Instruct, Qwen-3-4B-Instruct, and Phi-4-Mini show strong human-likeness and tone
- **Stage-based analysis reveals competitive performance in mid-stage interactions** where contextual reasoning is most critical
- Late-stage performance shows SLMs excel at resolution-focused responses
- Context summarization enables effective multi-turn dialogue handling with reduced context length

## Dataset

The dataset is constructed from the Customer Service Banking Conversation Corpus, processed through:
1. Multi-turn conversation construction
2. Context summarization using GPT-4o-mini
3. Response refinement using GPT-4.1
4. Content moderation filtering

**Dataset Statistics:**
- Training: 128,335 samples
- Validation: 18,333 samples  
- Test: 36,669 samples
- Average turns per conversation: ~10

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
- **Mid Stage (80%)**: Core interaction with substantive information exchange - requires strongest contextual reasoning
- **Late Stage (10%)**: Resolution and closure

### Stage-Based Insights
- **Early Stage**: Top SLMs show competitive issue identification (scores: 3.7-3.8)
- **Mid Stage**: LLaMA-3.1-8B and Qwen-3-8B remain competitive with commercial LLMs
- **Late Stage**: SLMs demonstrate strongest performance with resolution-focused responses (scores above 4.1)
- Stage-wise pairwise evaluation reveals SLMs are most competitive in mid-stage interactions where context maintenance is critical

This segmentation enables targeted analysis beyond overall performance scores, identifying which models excel at specific conversation phases.

## Training Configuration

- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Rank**: 16, Alpha: 32, Dropout: 0.1
- **Optimizer**: AdamW 8-bit
- **Learning Rate**: 2×10⁻⁵
- **Epochs**: 3
- **Max Sequence Length**: 512 tokens
- **Hardware**: NVIDIA RTX A100 40GB

## Citation

If you use this work, please cite:

```bibtex
@misc{slm_customer_service_qa,
  title={Can Small Language Models Handle Context-Summarized Multi-Turn Customer-Service QA? A Synthetic Data-Driven Comparative Evaluation},
  author={Research Team},
  year={2026}
}
```

## License

This project is intended for research purposes.

## Acknowledgments

We thank the human evaluators and the funding team for supporting API usage for large-scale model inference and evaluation.
