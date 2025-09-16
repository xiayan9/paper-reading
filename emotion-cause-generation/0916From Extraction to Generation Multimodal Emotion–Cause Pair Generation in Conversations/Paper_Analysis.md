# From Extraction to Generation: Multimodal Emotion–Cause Pair Generation in Conversations

- **Authors & Affiliation**: Heqing Ma, Jianfei Yu, Fanfan Wang, Hanyu Cao, Rui Xia  **School of Computer Science and Engineering, Nanjing University of Science and Technology**. 
- **Journal/ Year**: **IEEE Transactions on Affective Computing (T-AFFC),  2025**
- **JCR: Q1 （9.8）**
- Dataset: https://huggingface.co/datasets/NUSTM/ECGF/tree/main

---

## 1. Pain Points Identified

### 1.1 Implicit Causes are Missed
Traditional MECPE tasks focus on **extracting utterances** as emotion causes. This is too limited because:
- Many causes are **implicit**, requiring inference or world knowledge (e.g., sarcasm, unsaid reasons).
- Extraction cannot capture **abstract causal reasoning**.

*Example*: In *Friends*, Ross shows anger. The cause isn’t a literal utterance but Chandler’s sarcastic tone. Extraction-only models would fail.

### 1.2 Cross-Modal Triggers are Hard to Localize
- Many emotion triggers appear in **visual (facial expressions)** or **audio (prosody, pitch)** rather than text.  
- Utterance-level extraction is **too coarse**; it misses fine-grained cues like a smirk or stressed intonation.

*Example*: Monica smiles (visual) while making a cutting remark (text). The *smile* is the trigger of Rachel’s hurt feelings, but pure text analysis misses it.

## 2. Methods

### 2.1 New Task: MECPG
The authors introduce **Multimodal Emotion–Cause Pair Generation (MECPG)**:
- **Input**: a multimodal conversation (text + audio + video).
- **Output**: **(each non-neutral emotion utterance, emotion label, generated textual cause)**.

This task allows implicit and cross-modal causes to be **directly expressed** in natural language.

### 2.2 New Dataset: ECGF
- Built upon the Friends MECPE dataset.  
- For every **non-neutral utterance**, annotators **write a 1–3 sentence abstractive cause explanation**.  
- Annotators were allowed to infer hidden causes if consistent with video.  
- Three reviewers checked for quality (≈88% acceptance, Kappa > 0.6).

**Difference from older datasets**:

- Older MECPE corpora (e.g., ECF, RECCON) → *extractive*, labeling **utterances/spans**.  
- ECGF → *abstractive*, producing **short, human-readable summaries**.  

**Strengths vs. earlier corpora**:  

- *Older (extractive) strengths*: **high interpretability** at span-level (you see exactly which utterances were selected).  
- *ECGF strengths*:  
  - Captures **implicit** and **cross-modal** reasons that cannot be tied to a single utterance. 
  - **Uniform coverage**: every non-neutral emotion gets a cause explanation (rather than “None” when no specific cause utterance is found). 
  - **Better fit for generation models** and **human evaluation** of explanation quality.

*Example*:  
Old dataset: Cause = “Because Monica said ‘I hate you.’”  
ECGF: Cause = “Ross felt hurt because Monica mocked his job with a sarcastic smile.”

### 2.3 New Model: MONICA

A **BART-based sequence-to-sequence framework** with two strategies:

- **Pipeline**: Step 1 MERC (emotion recognition) → Step 2 cause generation.  
- **Joint**: MERC and cause generation trained together in one Seq2Seq decoder.  

**How multimodality is used**:
- **Text**: utterance embeddings from BART.  
- **Audio**: **HuBERT** features (prosodic and acoustic cues).  
- **Visual**: **C3D CNN** for facial/video frames.  
- Features are projected into the same space and fused before decoding.

**Generation process**:

- Decoder integrates context from all three modalities.  
- Produces a **natural language explanation** conditioned on multimodal evidence.

### 2.4 Limitations 
- **Multitask trade-off**: Joint training **helps MERC** but can **hurt generation**; T5 is **more sensitive** to loss weighting than BART. 
- **LLM fluency vs. hallucination**: GPT-3.5 looks **fluent** but introduces **fake information**, reducing contextual consistency in human eval.
- **Feature backbones**: visual **3D-CNN/C3D** lags modern **video transformers**; audio is better with **HuBERT**, but no instruction-aligned audio explored. 

---

### 2.5 Updated (2025) Remedies — with APA Evidence Under Each Bullet

**(A) Upgrade to “evidence-first, then explanation” (MECEC)**  

- **What**: **Locate evidence** (frames, acoustic segments, or text spans) → **conditioned generation** of a **longer, finer explanation** → **consistency checks** (evidence–explanation & emotion–cause).  
- **Why (2024–2025)**: The **MECEC** task and **ECEM** dataset emphasize that extractive paradigms are **too coarse** and early generative data (e.g., ECGF) gave **too-short** causes; **FAME-Net** reports strong results against multimodal/LLM baselines.  
  - **APA**:  
    - Wang, L., Yang, X., Feng, S., Wang, D., & Zhang, Y. (2025). *Generative Emotion Cause Explanation in Multimodal Conversations (MECEC).* In **ICMR 2025**. https://doi.org/10.1145/3731715.3733348 
    - Wang, L., Yang, X., Feng, S., Wang, D., & Zhang, Y. (2025, June 4). *Generative Emotion Cause Explanation in Multimodal Conversations.* **arXiv:2411.02430**. https://arxiv.org/abs/2411.02430 

**(B) Modernize visual/audio backbones + tighter cross-modal alignment**  

- **What**: Replace **C3D** with **VideoMAE** or **TimeSformer**; pursue **instruction-aligned** audio or stronger self-supervised audio, and **projection/distillation** to align with text semantics.  
- **Why (2025)**: MERC survey highlights **cross-modal alignment**, **missing/contradictory modalities**, and **efficiency** as bottlenecks; modern video pretraining is markedly stronger.  
  - **APA**:  
    - Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). **VideoMAE**: Masked autoencoders for self-supervised video pre-training. *NeurIPS 2022.* https://arxiv.org/abs/2203.12602 
    - Bertasius, G., Wang, H., & Torresani, L. (2021). **TimeSformer**. *ICML 2021.* https://arxiv.org/abs/2102.05095 
    - Wu, C., Cai, Y., Liu, Y., Zhu, P., Xue, Y., Gong, Z., Hirschberg, J., & Ma, B. (2025). *Multimodal Emotion Recognition in Conversations: A Survey…* **arXiv:2505.20511**. https://arxiv.org/abs/2505.20511 

**(C) Multi-objective scheduling + direct preference optimization (reduce hallucination)**  
- **What**: Decouple weights across **classification / evidence / explanation**, use **dynamic/uncertainty weighting**, and apply **DPO/RPO** with an **evidence-verifiability reward**.  
- **Why**: MECEC positions **explanation quality & semantic consistency** above pure n-gram, aligning with human judgments.  
  - **APA**: Wang, L., Yang, X., Feng, S., Wang, D., & Zhang, Y. (2025). *MECEC.* **ICMR 2025**. https://doi.org/10.1145/3731715.3733348 

**(D) Leverage video-LMM instruction tuning (mitigate modality gap)**  
- **What**: Distill/align with **LLaVA-Video** or similar as a *teacher* for video-text grounding and evidence localization.  
  - **APA**:  
    - Zhang, Y., Wu, J., Li, W., Ma, Z., Liu, Z., & Li, C. (2024). **LLaVA-Video**: Video instruction tuning with synthetic data. https://arxiv.org/abs/2410.02713 
    - LLaVA-VL Team. (2024, Sep 30). *LLaVA-Video: Video Instruction Tuning with Synthetic Data.* https://llava-vl.github.io/blog/2024-09-30-llava-video/ 

**(E) Evaluation upgrade (syntax → evidence consistency + human study)**  
- **What**: Keep BLEU/ROUGE/METEOR/CIDEr, BERTScore/BLEURT/SemSim, and **add** *evidence-consistency* and 3-axis **human evaluation** (Contextuality / Fluency / Relevance). Already adopted in MECEC.  
  - **APA**: Wang, L., Yang, X., Feng, S., Wang, D., & Zhang, Y. (2025). *MECEC.* **ICMR 2025**. https://doi.org/10.1145/3731715.3733348 

---

## 3 Contributions & Outlook

### 3.1 Paper’s Innovations
- **Task**: First systematic formulation of **MECPG** that unifies **emotion recognition + cause generation** to capture implicit and cross-modal triggers. :contentReference[oaicite:24]
- **Data**: **ECGF** provides **abstractive** causes per non-neutral utterance with **quality control** (pass-rate ≈ **88.3%**, Kappa > 0.6). 
- **Model**: **MONICA** (**BART**) unifies MERC + cause generation with multi-modal features and two training regimes. 

### 3.2 Outlook (paper + 2025)
- Improve **relevance & fair evaluation**, leverage **(M)LLMs** for perception, and push **cross-domain / cross-lingual** generalization with privacy in mind. 

---

## 4 Dataset: ECGF 

> **What is ECGF?** Built on ECF (Friends) MECPE corpus; for each **non-neutral** emotion utterance, annotators write a **1–3 sentence** abstractive cause summary. Neutral utterances receive “*The emotion Neutral has no causes*”. Reasonable inference is allowed when consistent with video evidence. 

![image-20250916160225320](From Extraction to Generation Multimodal Emotion–Cause Pair Generation in Conversations.assets/image-20250916160225320.png)

### 5 Metrics (for reproduction)
- **MERC metrics**: **Weighted_F1** and **Binary_F1** (to account for imbalance).

- **Generation metrics**: **BLEU-4 / METEOR / ROUGE-L / CIDEr** (syntactic) + **Sem-Sim / BLEURT / BERTScore** (semantic).

  | Metric      | Type             | Meaning                                                      | Characteristics                                              | Example Explanation                                          |
  | ----------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **BLEU4**   | Lexical match    | Measures overlap of 1–4 n-grams between generated and reference text | Sensitive to synonyms, focuses on surface overlap            | Output: "Ross got mad because Chandler mocked him." Reference: "Ross is angry because Chandler mocked him." → High score (phrase overlap) |
  | **METEOR**  | Lexical+semantic | Considers stemming, synonyms, and word order                 | Closer to human judgment, more tolerant than BLEU            | Output: "Ross was mad because Chandler made fun of him." Reference: "Ross is angry because Chandler mocked him." → Recognizes *mad~angry*, *made fun of~mocked* |
  | **ROUGE**   | Lexical match    | Measures overlapping segments, recall-oriented               | Common in summarization, emphasizes coverage                 | Output: "Ross felt angry because Chandler mocked him in front of friends." Reference: "Ross is angry because Chandler mocked him." → High score (covers key content) |
  | **CIDEr**   | Lexical+weight   | TF-IDF weighted n-gram overlap, emphasizes informative words | More sensitive to important keywords than BLEU               | Output includes "mocked", "angry" (high-weight words) → High score |
  | **Sem-Sim** | Semantic         | Computes semantic vector similarity between generated and reference | Ignores surface form, focuses on meaning                     | Output: "Ross is upset because Chandler teased him." Reference: "Ross is angry because Chandler mocked him." → High score (semantically close) |
  | **BLEURT**  | Semantic+quality | Pretrained LM fine-tuned on human ratings, predicts output quality | Can detect "fluent but wrong" outputs                        | Output: "Ross is angry because Chandler left." Reference: "Ross is angry because Chandler mocked him." → Low score (semantic error) |
  | **F_BERT**  | Semantic         | BERTScore-F1, measures token-level semantic alignment        | Balances coverage and accuracy, well-suited for explanations | Output: "Ross felt angry because Chandler teased him." Reference: "Ross was mad because Chandler mocked him." → High score (good token-level alignment) |

### 6 Main Results

![image-20250916160515172](From Extraction to Generation Multimodal Emotion–Cause Pair Generation in Conversations.assets/image-20250916160515172.png)

### 7 Ablation Study


![image-20250916160650637](From Extraction to Generation Multimodal Emotion–Cause Pair Generation in Conversations.assets/image-20250916160650637.png)
