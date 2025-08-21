# 📘 README.md

## Repository Structure

| **Folder**                  | **Files**                                                                                                                                                                          | **Description**                                                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `1-Central Bank BERT (MLM)` | `CentralBank-BERT.ipynb`                                                                                                                                                           | Domain-adaptive masked language model (pretraining on BIS speeches, 66M+ tokens). Produces `CentralBank-BERT`, the base for all downstream CBDC tasks. |
| `2-CBDC-BERT`               | `Evaluation/` (confusion, ROC) <br> `ML-Eval/` (baselines: Logistic, NB, RF, XGB) <br> `Sentence Bias Plots/` <br> `CBDC-BERT model training .ipynb` <br> `bert_training_data.csv` | Fine-tuned binary classifier (CBDC vs. Non-CBDC) built on CentralBank-BERT. Includes evaluation metrics vs. baselines and bias analysis.               |
| `3-CBDC-Stance`             | `cbdc-stance.ipynb` <br> `stance_sentences.csv`                                                                                                                                    | Classifies stance in CBDC discourse (Pro, Wait-and-See, Anti). Training data manually labeled from BIS speeches.                                       |
| `4-CBDC-Sentiment`          | `cbdc-sentiment.ipynb` <br> `cbdc_sentiment_training.csv`                                                                                                                          | Sentiment analysis on CBDC texts (Positive, Neutral, Negative).                                                                                        |
| `5-CBDC-Type`               | `cbdc-type.ipynb` <br> `cbdc_type_training.csv`                                                                                                                                    | Classifies CBDC mentions into Retail, Wholesale, or General/Unspecified.                                                                               |
| `6-CBDC-Discourse`          | `cbdc-discourse.ipynb` <br> `cbdc_classification_training.csv`                                                                                                                     | Categorizes CBDC sentences into Feature, Process, or Risk-Benefit.                                                                                     |

---

## Hugging Face Models

| **Model**                      | **Purpose (Description)**                                      | **Intended Use**                                         | **Model Link**                                                          |
| ------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| `bilalezafar/CentralBank-BERT` | Domain-adaptive masked LM trained on BIS speeches (1996–2024). | Base encoder for CBDC downstream tasks; fill-mask tasks. | [CentralBank-BERT](https://huggingface.co/bilalezafar/CentralBank-BERT) |
| `bilalezafar/CBDC-BERT`        | Binary classifier: CBDC vs Non-CBDC.                           | Flagging CBDC-related discourse in large corpora.        | [CBDC-BERT](https://huggingface.co/bilalezafar/CBDC-BERT)               |
| `bilalezafar/CBDC-Stance`      | 3-class stance model (Pro, Wait-and-See, Anti).                | Research on policy stances and discourse monitoring.     | [CBDC-Stance](https://huggingface.co/bilalezafar/CBDC-Stance)           |
| `bilalezafar/CBDC-Sentiment`   | 3-class sentiment (Positive, Neutral, Negative).               | Tone analysis in central bank communications.            | [CBDC-Sentiment](https://huggingface.co/bilalezafar/CBDC-Sentiment)     |
| `bilalezafar/CBDC-Type`        | Classifies Retail, Wholesale, General CBDC mentions.           | Distinguishing policy focus (retail vs wholesale).       | [CBDC-Type](https://huggingface.co/bilalezafar/CBDC-Type)               |
| `bilalezafar/CBDC-Discourse`   | 3-class discourse classifier (Feature, Process, Risk-Benefit). | Structured categorization of CBDC communications.        | [CBDC-Discourse](https://huggingface.co/bilalezafar/CBDC-Discourse)     |

---
* 📊 A **diagram image** (flow: CentralBank-BERT → downstream models),
* 📌 Or keep the README **purely tabular**?
