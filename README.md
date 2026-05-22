# Hate Speech Detection with Fine-Tuned Transformers

Fine-tuned and benchmarked three pretrained transformer architectures on **HateXplain** 
(19K annotated social media posts, 3-class) — DistilBERT, RoBERTa with class-weighted 
loss, and HateBERT (pretrained on 100M Reddit hate speech posts). Built a soft-voting 
ensemble achieving **70.1% accuracy and 0.691 macro F1**.

Extended to **Croatian** using multilingual BERT on the FRENK dataset as a cross-lingual 
transfer experiment.

## Models Compared
| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| TF-IDF Baseline | 0.6616 | 0.6525 |
| DistilBERT | 0.6959 | 0.6861 |
| RoBERTa (Weighted) | 0.6772 | 0.6681 |
| HateBERT | 0.6949 | 0.6848 |
| **Ensemble** | **0.7011** | **0.6911** |

## Files
- `TRIA_final.ipynb` — full pipeline: EDA, preprocessing, fine-tuning, evaluation
- `TRIA_Report.pdf` — project report with results and analysis

## Stack
Python · PyTorch · Hugging Face Transformers · Scikit-learn · Google Colab