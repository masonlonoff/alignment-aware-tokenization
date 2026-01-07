# Alignment-Aware Tokeniation

This repository contains the final project report and reference materials for **Alignment-Aware Tokenization (AAT)**, a research project exploring how subword tokenization can introduce unintended alignment and safety effects in large language models. The work investigates how hazardous morphemes can "spill over" into benign contexts through shared subword fragments, leading to elevated hazard activation even when model inputs are neutral. 

This project was completed as a final course project and presents novel research contributions, rather than a production system or fully packaged library. 

**Project Report**:
['alignment_aware_tokenization.pdf'](./alignment_aware_tokenization.pdf)


## Research Motivation

Subword tokenizers are typically evaluated on compression and language modeling efficiency. This project asks a different question:

**Can tokenization itself influence alignment behavior by coupling hazardous concepts to benign text via shared subwords?**

We refer to this phenomenon as **subword spillover** and study its impact on safety-relevant representations inside transformer models.


## High-Level Approach

This project introduces **Alignment-Aware Tokenization (AAT)**, a framework that combines:

- **Hazard concept probing**
  - A mid-layer representation probe trained with a small number of labeled and neutral anchor examples to estimate a "hazard direction" in activation space.

- **Drift-regularized fine-tuning**
  - Lightweight LoRA adaptation that penalizes elevated hazard activation on neutral text, encouraging representational stability.

- **Tokenizer-aware interventions**
  - *BPE models*: targeted merge pruning to reduce hazardous subword reuse
  - *Unigram (SentencePiece) models*: hazard-aware priors to discourage spillover substrings

The emphasis is on conceptual alignment effects, not on maximizing benchmark performance. 

## Key Findings

- Tokenization choices can meaningfully affect safety-relevant internal representations, even without changing model architecture.
- Hazard probes trained with very few labels saturate quickly, making them practical for low-resource alignment analysis.
- For Unigram tokenizers, tokenizer-only changes can strongly mismatch pretrained weights, highlighting the need for paired adaptation.
- Alignment behavior emerges from interactions between tokenization, representation learning, and fine-tuning**, not from any single component.

Detailed quantitative results and analysis are provided in the project report.


## Authors & Collaboration

This project was completed collaboratively by:

- **Dipesh Tharu Mahato**  
- **Ankit Chahar**  
- **Evan Beck**  
- **Mason Lonoff**  
- **Varshitha Reddy Medarametla**

All authors contributed to the research design, experimentation, and analysis.

## Original Repository

The primary codebase and experimental development were conducted in the original group repository:

https://github.com/dipeshbabu/alignment-aware-tokenization

## Disclaimer

This repository is intended for academic reference and portfolio purposes. It is not maintained as a standalone research library and is not optimized for external reproduction.

