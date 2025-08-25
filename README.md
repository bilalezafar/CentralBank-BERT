
# 📘 Central Bank Digital Currency (CBDC) – NLP Models and Pipelines

This repository contains a suite of **transformer-based models and training pipelines** designed for the analysis of Central Bank Digital Currency (CBDC) discourse. Built on top of a domain-adapted BERT model [Central Bank BERT](https://huggingface.co/bilalzafar/CentralBank-BERT), these pipelines enable classification of CBDC-related text into multiple dimensions such as **CBDC detection, stance, sentiment, type, and discourse features**. The project covers **end-to-end workflows** including:

* **Data preprocessing** and annotation files,
* **Fine-tuning pipelines** for each classification task,
* **Evaluation scripts and visualizations**,
* **Released Hugging Face models** for downstream deployment.

The goal is to provide a reproducible and extensible resource for **policy researchers, economists, and NLP practitioners** studying CBDCs in speeches, reports, and media articles.

---

## 🔹 Repository Structure

| **Folder**                  | **Files & Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-Central Bank BERT (MLM)` | `CentralBank-BERT.ipynb` — Notebook for **domain-adaptive pretraining** (Masked Language Modeling). Uses \~2,000 BIS speeches (66M+ tokens). Outputs the **[CentralBank-BERT](https://huggingface.co/bilalzafar/CentralBank-BERT)** model.                                                                                                                                                                                                                                                                                                     |
| `2-CBDC-BERT`               | `CBDC speeches data extraction for BERT.ipynb` - Pipline for data extraction for CBDC-BERT model training. <br> `cbdc_keywords.csv` - CBDC related keywords. <br> `CBDC-BERT model training.ipynb` — Fine-tuning pipeline (**CBDC vs Non-CBDC**) with tokenization, training loop, and evaluation. <br> `bert_training_data.csv` — Training dataset: CBDC class = 5,390 sentences; Non-CBDC = 5,610 sentences. <br> `Evaluation/` — Confusion matrices, ROC-AUC plots, and per-class metrics. <br> `ML-Eval/` — Classical baselines (LogReg, Naive Bayes, RF, XGBoost). <br> `Sentence Bias Plots/` — Visualization of sentence-level prediction biases. Output Model: **[CBDC-BERT](https://huggingface.co/bilalzafar/CBDC-BERT)** |
| `3-CBDC-Stance`             | `cbdc-stance.ipynb` — Pipeline for **stance classification** (Pro, Wait-and-See, Anti). Includes preprocessing, fine-tuning, evaluation. <br> `stance_sentences.csv` — Annotated dataset (n≈1,647), balanced across three stance labels. Output Model: **[CBDC-Stance](https://huggingface.co/bilalzafar/CBDC-Stance)**                                                                                                                                                                                                                                               |
| `4-CBDC-Sentiment`          | `cbdc-sentiment.ipynb` — Pipeline for **sentiment classification** (Positive, Neutral, Negative). Includes tokenization, class weighting, evaluation. <br> `cbdc_sentiment_training.csv` — Annotated dataset (n≈2,065), balanced with downsampling. Output Model: **[CBDC-Sentiment](https://huggingface.co/bilalzafar/CBDC-Sentiment)**                                                                                                                                                                                                                                    |
| `5-CBDC-Type`               | `cbdc-type.ipynb` — Pipeline for **CBDC type classification** (Retail, Wholesale, General). Weighted balancing for smaller wholesale class. <br> `cbdc_type_training.csv` — Dataset (n≈1,417; Retail 543 / Wholesale 228 / General 646). Output Model: **[CBDC-Type](https://huggingface.co/bilalzafar/CBDC-Type)**                                                                                                                                                                                                                                               |
| `6-CBDC-Discourse`          | `cbdc-discourse.ipynb` — Pipeline for **CBDC discourse categorization** (Feature, Process, Risk-Benefit). Unweighted training, class metrics. <br> `cbdc_classification_training.csv` — Dataset (n=2,886; 962 per class). Output Model: **[CBDC-Discourse](https://huggingface.co/bilalzafar/CBDC-Discourse)**                                                                                                                                                                                                                                                              |
| `7-CentralBank-NER`         | `CentralBank-NER.ipynb` — Pipeline for **Named Entity Recognition (NER)** in central banking discourse. Built on domain-adapted `CentralBank-BERT`. <br> `Access link to BIS Speeches NER dataset.docx` — Dataset access instructions. Output Model: **[CentralBank-NER](https://huggingface.co/bilalzafar/CentralBank-NER)**                                                                                                                                                                                                                                                 |
| `8-CBDC-Analysis and Results` | End-to-end notebook (**CBDC Analysis and Results.ipynb**) that reads **cbdc-dataset-final.csv**, runs all CBDC classifiers to create **cbdc-dataset-predictions.csv**, and generates all manuscript tables and figures; outputs saved to **prediction\_results/** and **descriptive\_results/**. |

---

## 🤗 Hugging Face Models

| Model                          | Purpose                                                             | Intended Use                                                                         | Model Link                                                              |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `bilalezafar/CentralBank-BERT` | Domain-adaptive masked LM trained on BIS speeches (1996–2024).      | Base encoder for CBDC downstream tasks; fill-mask tasks.                             | [CentralBank-BERT](https://huggingface.co/bilalzafar/CentralBank-BERT) |
| `bilalezafar/CBDC-BERT`        | Binary classifier: CBDC vs Non-CBDC.                                | Flagging CBDC-related discourse in large corpora.                                    | [CBDC-BERT](https://huggingface.co/bilalzafar/CBDC-BERT)               |
| `bilalezafar/CBDC-Stance`      | 3-class stance model (Pro, Wait-and-See, Anti).                     | Research on policy stances and discourse monitoring.                                 | [CBDC-Stance](https://huggingface.co/bilalzafar/CBDC-Stance)           |
| `bilalezafar/CBDC-Sentiment`   | 3-class sentiment (Positive, Neutral, Negative).                    | Tone analysis in central bank communications.                                        | [CBDC-Sentiment](https://huggingface.co/bilalzafar/CBDC-Sentiment)     |
| `bilalezafar/CBDC-Type`        | Classifies Retail, Wholesale, General CBDC mentions.                | Distinguishing policy focus (retail vs wholesale).                                   | [CBDC-Type](https://huggingface.co/bilalzafar/CBDC-Type)               |
| `bilalezafar/CBDC-Discourse`   | 3-class discourse classifier (Feature, Process, Risk-Benefit).      | Structured categorization of CBDC communications.                                    | [CBDC-Discourse](https://huggingface.co/bilalzafar/CBDC-Discourse)     |
| `bilalezafar/CentralBank-NER`  | Named Entity Recognition (NER) model for central banking discourse. | Identifying institutions, persons, and policy entities in BIS/central bank speeches. | [CentralBank-NER](https://huggingface.co/bilalzafar/CentralBank-NER)   |


---

## 📖 Citation  

If you use this repository or models in your research, please cite as:  

**Zafar, M. B. (2025). *CentralBank-BERT: Machine Learning Evidence on Central Bank Digital Currency Discourse*. SSRN. [https://papers.ssrn.com/abstract=5404456](https://papers.ssrn.com/abstract=5404456)**

**BibTeX**

```bibtex
@article{zafar2025centralbankbert,
  title={CentralBank-BERT: Machine Learning Evidence on Central Bank Digital Currency Discourse},
  author={Zafar, Muhammad Bilal},
  year={2025},
  journal={SSRN Electronic Journal},
  url={https://papers.ssrn.com/abstract=5404456}
}
```
