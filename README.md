# Semi-Structured Data Analysis: Text Classification & Topic Modelling

FIT5212 Assignment 1 (Monash University). The notebook works on a corpus of research-paper titles and abstracts in two parts:

1. **Text classification:** predict whether a paper belongs to *Computational Linguistics*.
2. **Topic modelling:** discover latent topics in abstracts with Latent Dirichlet Allocation (LDA).

**Author:** Akshat Dawange (Student ID: 34611266)

## 🔍 Part 1: Text Classification

**Target label:** `ComputationalLinguistics` (binary). The training set also contains `MachineLearning` and `HumanComputerInteraction` labels, which are used only for the class-distribution plot.

### Pipeline
1. **Pre-processing:** lowercase, remove non-letters, tokenise, remove stop words and tokens of 2 characters or fewer, then Porter-stem. This produces `clean_title` and `clean_abstract`.
2. **Features:**
   - Logistic Regression uses TF-IDF with 5,000 features.
   - RNN uses a Keras tokenizer and padded sequences. Abstracts use a vocabulary of 10,000 words and length 200. Titles use a vocabulary of 5,000 words and length 30.
3. **Models:**
   - Logistic Regression
   - RNN (Embedding → LSTM(64) → Dropout(0.3) → Sigmoid)
   - Extra baselines: Linear SVM and Multinomial Naive Bayes
4. **Evaluation:** accuracy, precision, recall, F1 and precision-recall curves on the dev set, plus a final run on the test set.

### Experimental configurations
Eight configurations vary three things: **input** (title or abstract), **model** (Logistic Regression or RNN) and **training size** (first 1,000 rows or the full training set).

### Dev-set results (27,132 papers)

| Model | Input | Training size | Accuracy |
|---|---|---|---|
| Logistic Regression | Abstract | 1,000 | 0.7818 |
| Logistic Regression | Abstract | Full | 0.8842 |
| Logistic Regression | Title | 1,000 | 0.7512 |
| Logistic Regression | Title | Full | 0.8606 |
| RNN | Title | 1,000 | 0.7867 |
| RNN | Title | Full | 0.8660 |
| Linear SVM | Abstract | Full | 0.8849 |
| Naive Bayes | Abstract | Full | 0.8740 |

**Takeaways**
- More training data helps every model. Logistic Regression on abstracts gains about 10 points of accuracy, from 0.78 to 0.88.
- Abstracts beat titles, which suggests the extra text gives the model more to work with.
- With only 1,000 training rows, Logistic Regression has high recall (about 0.98) but low recall for the negative class. It is mostly predicting the majority class.
- Linear SVM scored highest of the models listed above.

## 🧩 Part 2: Topic Modelling (LDA)

Built with Gensim on the cleaned abstracts.

| Config | Tokens | Topics (K) | Documents |
|---|---|---|---|
| A1 | Unigrams | 10 | 1,000 |
| A2 | Unigrams | 10 | 20,000 |
| B1 | Bigrams | 40 | 1,000 |
| B2 | Bigrams | 40 | 20,000 |

- Settings: `passes=10`, `alpha='auto'`, `random_state=42`.
- Bigrams come from Gensim `Phrases` (`min_count=5`, `threshold=10`).
- Outputs are top-term lists, bar charts of the top 10 words per topic, and word clouds.
- The notebook's reflection says **B2 (bigrams, 20,000 documents)** gave the best topics.

## ⚙️ Setup

```bash
pip install pandas numpy scikit-learn nltk tqdm seaborn matplotlib \
            tensorflow gensim wordcloud pyLDAvis
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Data
The notebook expects three CSV files: `train_set.csv`, `dev_set.csv` and `test_set.csv`. They are not included in this repo because they are course data. Each needs `title`, `abstract` and the label columns. Update the file paths in the data-loading cell, which currently point to a local Windows directory.

## ⚠️ Notes
- Data paths are hard-coded for a local machine.
- Some cells reuse variable names such as `model`, `vectorizer` and `X_train`. The test-set evaluation at the end uses whichever model was trained last, so run the notebook top to bottom.
- The `pyLDAvis` visualisations are mentioned in the notebook but need the libraries installed.
