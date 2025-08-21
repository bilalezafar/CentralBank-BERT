
# 📘 Central Bank Digital Currency (CBDC) – NLP Models and Pipelines

This repository contains a suite of **transformer-based models and training pipelines** designed for the analysis of Central Bank Digital Currency (CBDC) discourse. Built on top of a domain-adapted BERT model [CentralBank-BERT](https://huggingface.co/bilalzafar/CentralBank-BERT), these pipelines enable classification of CBDC-related text into multiple dimensions such as **CBDC detection, stance, sentiment, type, and discourse features**. The project covers **end-to-end workflows** including:

* **Data preprocessing** and annotation files,
* **Fine-tuning pipelines** for each classification task,
* **Evaluation scripts and visualizations**,
* **Released Hugging Face models** for downstream deployment.

The goal is to provide a reproducible and extensible resource for **policy researchers, economists, and NLP practitioners** studying CBDCs in speeches, reports, and media articles.

---

## 🔹 Repository Structure

| **Folder**                  | **Files**                          | **Description**                                                                                                                                                                                                  |
| --------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-Central Bank BERT (MLM)` | `CentralBank-BERT.ipynb`           | Notebook for **domain-adaptive pretraining** (Masked Language Modeling). Uses \~2,000 BIS speeches (66M+ tokens). Outputs the `CentralBank-BERT` model.                                                          |
| `2-CBDC-BERT`               | `CBDC-BERT model training.ipynb`   | Fine-tuning pipeline (binary classification: **CBDC vs Non-CBDC**) with tokenization, training loop, and evaluation. <br> **Inputs**: `bert_training_data.csv`. <br> **Outputs**: fine-tuned Hugging Face model. |
|                             | `bert_training_data.csv`           | Training dataset: The CBDC class contains 5,390 sentences, and the Non-CBDC class contains 5,610 sentences.                                                                                                               |
|                             | `Evaluation/`                      | Contains confusion matrices, ROC-AUC plots, and per-class performance metrics.                                                                                                                                   |
|                             | `ML-Eval/`                         | Classical baselines (LogReg, Naive Bayes, RF, XGBoost) for comparison with BERT.                                                                                                                                 |
|                             | `Sentence Bias Plots/`             | Visualization of sentence-level prediction biases (e.g., over-predicting CBDC terms).                                                                                                                            |
| `3-CBDC-Stance`             | `cbdc-stance.ipynb`                | Pipeline for **stance classification** (Pro, Wait-and-See, Anti). Includes data preprocessing, model fine-tuning, and evaluation.                                                                                |
|                             | `stance_sentences.csv`             | Annotated training data (n≈1,647). Balanced across three stance labels.                                                                                                                                          |
| `4-CBDC-Sentiment`          | `cbdc-sentiment.ipynb`             | Pipeline for **sentiment classification** (Positive, Neutral, Negative). Includes tokenization, training with class weights, and evaluation.                                                                     |
|                             | `cbdc_sentiment_training.csv`      | Annotated dataset (n≈2,065 sentences). Balanced with downsampling for training.                                                                                                                                  |
| `5-CBDC-Type`               | `cbdc-type.ipynb`                  | Pipeline for **CBDC type classification** (Retail, Wholesale, General). Implements weighted class balancing for smaller wholesale set.                                                                           |
|                             | `cbdc_type_training.csv`           | Dataset (n≈1,417 sentences; Retail 543 / Wholesale 228 / General 646).                                                                                                                                           |
| `6-CBDC-Discourse`          | `cbdc-discourse.ipynb`             | Pipeline for **CBDC discourse categorization** (Feature, Process, Risk-Benefit). Includes unweighted training and evaluation with class metrics.                                                                 |
|                             | `cbdc_classification_training.csv` | Training dataset 2,886 sentences (962 per class); Feature / Process / Risk-Benefit.                                                                                                                    |

---

## 🤗 Hugging Face Models

| **Model**                      | **Purpose**                                      | **Intended Use**                                         | **Model Link**                                                          |
| ------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| `bilalezafar/CentralBank-BERT` | Domain-adaptive masked LM trained on BIS speeches (1996–2024). | Base encoder for CBDC downstream tasks; fill-mask tasks. | [CentralBank-BERT](https://huggingface.co/bilalzafar/CentralBank-BERT) |
| `bilalezafar/CBDC-BERT`        | Binary classifier: CBDC vs Non-CBDC.                           | Flagging CBDC-related discourse in large corpora.        | [CBDC-BERT](https://huggingface.co/bilalzafar/CBDC-BERT)               |
| `bilalezafar/CBDC-Stance`      | 3-class stance model (Pro, Wait-and-See, Anti).                | Research on policy stances and discourse monitoring.     | [CBDC-Stance](https://huggingface.co/bilalzafar/CBDC-Stance)           |
| `bilalezafar/CBDC-Sentiment`   | 3-class sentiment (Positive, Neutral, Negative).               | Tone analysis in central bank communications.            | [CBDC-Sentiment](https://huggingface.co/bilalzafar/CBDC-Sentiment)     |
| `bilalezafar/CBDC-Type`        | Classifies Retail, Wholesale, General CBDC mentions.           | Distinguishing policy focus (retail vs wholesale).       | [CBDC-Type](https://huggingface.co/bilalzafar/CBDC-Type)               |
| `bilalezafar/CBDC-Discourse`   | 3-class discourse classifier (Feature, Process, Risk-Benefit). | Structured categorization of CBDC communications.        | [CBDC-Discourse](https://huggingface.co/bilalzafar/CBDC-Discourse)     |

---

## 📖 Citation  

If you use this repository or models in your research, please cite as:  

Zafar, M. B. (2025). *Central Bank Digital Currency Discourse Analysis with Domain-Adapted BERT Models*.  
