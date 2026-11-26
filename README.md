## Project overview

This project builds an NLP pipeline to classify medical research abstracts into one of five medical-condition categories. It investigates whether a self-trained embedding or a pretrained biomedical embedding provides better representations for sequential text classification, evaluates the embeddings against a biomedical semantic-similarity benchmark, and tests whether simple features from biomedical named entity recognition (NER) improve classification.

The work is implemented primarily with Python, Gensim, NLTK, scikit-learn and PyTorch. The sequential classifier is a Bidirectional Long Short-Term Memory network (Bi-LSTM).

## Dataset

The notebook uses the assignment-provided medical abstract dataset:

- **Training set:** 11,550 abstracts.
- **Test set:** 2,883 abstracts, held out for final evaluation.
- **Validation set:** 2,310 abstracts (20% of the training set), created with a stratified split and a fixed random state.
- **Labels:** five medical-condition categories.
- **Intrinsic evaluation:** 449 concept pairs from UMNSRS.

The five class labels are:

| Label | Medical condition |
|---:|---|
| 1 | Neoplasms |
| 2 | Digestive system diseases |
| 3 | Nervous system diseases |
| 4 | Cardiovascular diseases |
| 5 | General pathological conditions |

## Methodology and work completed

### 1. Text preprocessing and self-trained embeddings

Medical abstracts were sentence-tokenised, converted to lowercase, cleaned by removing punctuation and non-alphanumeric characters, word-tokenised and lemmatised with NLTK WordNet. Stopwords were retained and stemming was not applied so that potentially useful medical information was not discarded.

The training abstracts produced **99,184 sentences and 2,143,214 tokens** after preprocessing. Five embedding configurations were explored:

| Experiment | Method | Vector size | Window | Minimum count | Epochs |
|---|---|---:|---:|---:|---:|
| 1 | Word2Vec CBOW | 100 | 5 | 5 | 20 |
| 2 | Word2Vec Skip-gram | 200 | 10 | 5 | 15 |
| 3 | Word2Vec CBOW | 200 | 10 | 2 | 30 |
| 4 | FastText Skip-gram | 300 | 20 | 3 | 10 |
| 5 | FastText CBOW | 100 | 10 | 2 | 20 |

Nearest-neighbour examples were inspected for terms such as *cancer* and *tumor*. The **300-dimensional FastText Skip-gram** model was selected as the self-trained embedding for the later intrinsic and extrinsic comparisons. Its subword representation was considered useful for handling morphological variation in biomedical terminology.

### 2. Pretrained biomedical embedding

The pretrained embedding selected was **BioWordVec**, specifically the 200-dimensional `bio_embedding_extrinsic` vectors described by Zhang et al. (2019). The notebook reports a vocabulary of **2,324,849 entries**. BioWordVec was loaded in word2vec binary format using Gensim `KeyedVectors`.

The embedding was selected because it combines subword information with biomedical resources and PubMed/MeSH-derived information, making it relevant to medical abstracts.

### 3. Intrinsic evaluation with UMNSRS

Cosine similarity was calculated for concept pairs and compared with the UMNSRS human ratings using Spearman's rank correlation. The notebook reports **207 usable pairs** for the comparison:

| Embedding | Spearman's ρ | p-value |
|---|---:|---:|
| FastText Skip-gram (300d) | 0.5735 | 1.68 × 10⁻¹⁹ |
| BioWordVec (200d) | **0.6349** | 9.28 × 10⁻²⁵ |

BioWordVec had the higher correlation, indicating that its similarity rankings were more closely aligned with the UMNSRS human ratings on the evaluated pairs.

**Reproducibility note:** before publishing, review the pair-filtering code in the UMNSRS cell. It filters vocabulary membership using `model_2` (the Word2Vec Skip-gram model) and BioWordVec, but then calculates the self-trained similarities using `model_4` (FastText Skip-gram). The accompanying discussion describes the filter as using the FastText vocabulary. Align the code and explanation, then rerun this task if you want the reported pair-selection procedure to match exactly.

### 4. Medical abstract classification

A stratified 80/20 split of the 11,550 training abstracts created 9,240 training examples and 2,310 validation examples. The vocabulary was built from the training split, with `<PAD>` and `<UNK>` tokens and a minimum word frequency of two. Abstracts were converted to integer sequences and padded or truncated to **200 tokens**. The resulting vocabulary contained **22,050 entries**.

The classifier is a PyTorch **Bidirectional LSTM (Bi-LSTM)**. It uses a pretrained/self-trained embedding matrix, a bidirectional LSTM, max pooling over the sequence outputs, dropout and a fully connected output layer for the five classes. The embedding weights were frozen during classifier training.

Main training settings:

| Setting | Value |
|---|---:|
| Hidden dimension | 64 |
| Dropout | 0.5 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 20 |
| Early-stopping patience | 2 epochs |
| Loss | Cross-entropy |

Both embedding-based classifiers used the same architecture and training settings. The test-set results saved in the notebook are:

| Embedding / classifier | Test accuracy | Weighted F1 |
|---|---:|---:|
| FastText + Bi-LSTM | 62.33% | 61.31% |
| **BioWordVec + Bi-LSTM** | **64.13%** | **63.00%** |

BioWordVec exceeded FastText by **1.80 percentage points in accuracy** and **1.69 percentage points in weighted F1**. The notebook's per-class reports and confusion matrices show that class 5 (general pathological conditions) was particularly difficult for both models.

The embedding matrix covered the full 22,050-word classifier vocabulary for FastText in the notebook's reported check. BioWordVec had 131 words without a direct vector; these entries were assigned randomly in the embedding matrix.

### 5. Biomedical NER features

The pretrained scispaCy model **`en_ner_bc5cdr_md`**, based on the BC5CDR corpus, was applied to abstracts. It identifies two entity types: `DISEASE` and `CHEMICAL`. The notebook includes example extracted entities, such as disease names and chemical mentions.

For each abstract, the number of detected disease entities and chemical entities was counted. These two numerical features were concatenated with the max-pooled Bi-LSTM representation before the classifier's final layer.

The BioWordVec baseline was compared with the NER-augmented model on the same test set:

| Model | Test accuracy | Weighted F1 |
|---|---:|---:|
| BioWordVec + Bi-LSTM (no NER) | **64.13%** | **63.00%** |
| BioWordVec + Bi-LSTM + NER counts | 62.89% | 62.26% |

The count-based NER features did **not** improve performance. The NER-enhanced model was lower by 1.24 percentage points in accuracy and 0.74 percentage points in weighted F1. A plausible explanation discussed in the notebook is that reducing all extracted entities to two counts discards their identities and contextual relationships; NER extraction errors may also introduce noisy features.

### 6. Inference pipeline

The final notebook section contains a Google Colab input interface where a user can paste an unseen medical abstract and request a predicted condition. The selected model is the **BioWordVec Bi-LSTM without NER**, because it achieved the strongest test-set results among the evaluated classifiers.

To make the inference section independent of the earlier training cells, the notebook embeds a compressed model package containing the model configuration, vocabulary, quantised embedding matrix and trained LSTM/classifier weights. The inference code reconstructs the model, applies the same lowercasing, cleaning, tokenisation and lemmatisation, maps unknown tokens to `<UNK>`, pads/truncates to 200 tokens, and displays the predicted condition and its softmax confidence.

## Main findings

1. BioWordVec performed better than the self-trained FastText embedding in both the UMNSRS intrinsic evaluation and the medical-abstract classification test.
2. The BioWordVec Bi-LSTM was the strongest evaluated classifier, with **64.13% test accuracy** and **63.00% weighted F1**.
3. Adding simple counts of disease and chemical entities from scispaCy did not improve the BioWordVec classifier.
4. General pathological conditions (class 5) remained a challenging category, indicating that the model's performance was not uniform across all classes.

Common libraries used include Python, pandas, NumPy, NLTK, Gensim, SciPy, scikit-learn, PyTorch, spaCy and ipywidgets. 

## Methodological notes and limitations

- The notebook trains its self-trained embeddings on the full supplied training dataset before making the classifier's 80/20 training/validation split. The held-out test set is not used to train the embeddings or classifier. However, the validation abstracts are present in the unsupervised embedding corpus; for a strictly isolated validation experiment, retrain the self-trained embedding using only the 9,240 classifier-training abstracts.
- The classifier uses fixed embedding weights rather than fine-tuning the embeddings during supervised training.
- Abstracts are limited to 200 tokens, so information beyond that limit is truncated.
- The NER experiment uses only two entity-count features, not entity identities, relations or contextual spans.
- The UMNSRS comparison uses a subset of the 449 supplied pairs because vocabulary coverage is required. See the reproducibility note in Task 3 and align the filtering implementation with the stated method before reporting the correlation as a final result.
- The inference pipeline is implemented in the notebook, but anyone publishing it should independently test the final cell in a clean Colab runtime.

## External resources

- **BioWordVec:** Zhang et al. (2019), *Improving Biomedical Word Embeddings with Subword Information and MeSH Ontology*. [Figshare dataset and vectors](https://figshare.com/articles/dataset/Improving_Biomedical_Word_Embeddings_with_Subword_Information_and_MeSH_Ontology/6882647?file=12551780).
- **scispaCy:** Neumann, M., King, D., Beltagy, I. & Ammar, B. (2019), *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*. [GitHub repository](https://github.com/allenai/scispacy) · [paper](https://arxiv.org/abs/1902.07669).
- **UMNSRS:** [BigBio UMNSRS dataset](https://huggingface.co/datasets/bigbio/umnsrs).
