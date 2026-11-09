# IMDb Review Sentiment Analysis

Classify English movie reviews as **positive** or **negative** using TF-IDF and machine learning. The notebook covers data inspection, preprocessing, training, evaluation, and model comparison. A Streamlit app lets users enter their own reviews and view predictions.

## Features

- Inspect missing values, class distribution, and duplicate reviews.
- Clean HTML, punctuation, whitespace, and contractions.
- Remove stopwords while preserving `no`, `not`, `nor`, and `never`.
- Extract TF-IDF features using unigrams and bigrams.
- Train a Logistic Regression baseline and tune its regularization with cross-validation.
- Compare Logistic Regression with Multinomial Naive Bayes.
- Display classification reports, confusion matrices, and frequent-word charts.
- Predict custom reviews in a Streamlit interface with example inputs and model probability scores.

## Dataset

The project uses `data/IMDB Dataset.csv`, containing 50,000 English movie reviews: 25,000 positive and 25,000 negative. The expected columns are `review` and `sentiment`.

After preprocessing and duplicate removal, 49,575 reviews remain. A stratified split with `random_state=40` assigns 39,660 reviews to training and 9,915 to testing.

IMDb is one of the dataset options listed in the assignment. This project evaluates movie reviews, not general product reviews.

## Project Structure

```text
Sentiment-Analysis/
├── data/
│   └── IMDB Dataset.csv
├── sentiment_analysis.ipynb   # Training, evaluation, and visualizations
├── sentiment_model.joblib     # Saved TF-IDF and tuned classifier pipeline
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

Developed with Python 3.12.10. Run the following commands from the project folder in PowerShell.

Create a virtual environment if one does not already exist:

```powershell
python -m venv .venv-sentiment
```

Activate it and install dependencies:

```powershell
.\.venv-sentiment\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Run the Notebook

1. Place the IMDb CSV at `data/IMDB Dataset.csv`.
2. Open `sentiment_analysis.ipynb` in VS Code or Jupyter.
3. Select the `.venv-sentiment` Python kernel.
4. Run the cells from top to bottom. Cross-validation takes longer than the other steps.
5. Run the final save cell to create or replace `sentiment_model.joblib` in the project folder.

The save cell uses:

```python
import joblib

joblib.dump(best_model, "sentiment_model.joblib")
```

`best_model` must be defined by running the tuning and tuned evaluation cells first.

## Run the Streamlit App

Ensure `sentiment_model.joblib` is beside `app.py`, then run:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Enter an English movie review or select an example, then click **Analyze sentiment**. The app preprocesses the review using the same rules as the notebook before passing it to the saved pipeline. It does not retrain the model.

The Clear button resets the input and result. Editing a review hides its previous prediction until it is analyzed again.

## Public deployment

The app runs with Python 3.12. Runtime dependencies in `requirements.txt` match
the environment used to verify the saved model. Notebook dependencies are in
`requirements-dev.txt`; they are not needed to serve predictions.

1. Create a public GitHub repository and upload `app.py`,
   `sentiment_model.joblib`, `requirements.txt`, `requirements-dev.txt`,
   `sentiment_analysis.ipynb`, `.gitignore`, and `README.md`.
2. Sign in at https://share.streamlit.io/ with GitHub and select **Create app**.
3. Choose the repository, branch `main`, and entrypoint `app.py`.
4. Under **Advanced settings**, select Python **3.12**, then deploy.
5. Set app sharing to public and test the resulting URL while signed out.

The training dataset is excluded from GitHub by `.gitignore`. The deployed app
uses the saved model and does not need the dataset. Keep virtual environments,
`.env` files, and Streamlit secrets out of the repository.

Deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Method

1. Load and inspect the reviews.
2. Remove missing records and duplicate reviews.
3. Normalize text and remove stopwords, empty reviews, and duplicate cleaned reviews.
4. Split the data into 80% training and 20% testing.
5. Fit TF-IDF on training data only, with up to 50,000 features, `ngram_range=(1, 2)`, and `sublinear_tf=True`.
6. Train and evaluate baseline Logistic Regression.
7. Search `C` values `[0.1, 1, 3, 10]` using three-fold stratified cross-validation on training data. A pipeline fits TF-IDF separately within each fold.
8. Evaluate the selected model on the test set and compare it with Multinomial Naive Bayes.
9. Plot the 20 most frequent words in positive and negative training reviews.
10. Save the tuned pipeline for the app.

## Results

Recorded notebook results on the same test set:

| Model | Test Accuracy |
|---|---:|
| Baseline Logistic Regression | 90.29% |
| Tuned Logistic Regression | **90.55%** |
| Multinomial Naive Bayes | 88.05% |

Cross-validation selected **C=10**, with a mean validation accuracy of **90.04%**. The tuned model improved test accuracy by **0.26 percentage points** over the baseline.

| Tuned Model Metric | Negative | Positive |
|---|---:|---:|
| Precision | 90.95% | 90.16% |
| Recall | 89.98% | 91.12% |
| F1-score | 90.46% | 90.63% |

The bonus tasks are included: Naive Bayes accuracy comparison and word-frequency visualization by sentiment. Frequent words indicate usage within each class; they are not necessarily sentiment-specific or explanations of an individual prediction.

## Tools

Python, Pandas, scikit-learn, Matplotlib, Joblib, and Streamlit are used directly. Text normalization uses Python's `re` and `html` modules; stopwords come from scikit-learn. Other packages listed in `requirements.txt`, including NLTK, are installed dependencies but are not currently used in the implementation.

## Limitations

- Results are based on one IMDb train/test split and do not establish performance on other datasets.
- Bengali reviews and product reviews have not been evaluated.
- Sarcasm, mixed opinions, and context-dependent language may lead to incorrect predictions.
- The model supports two labels; it does not predict a neutral class.
- Probability scores are model estimates, not guaranteed confidence or accuracy for an individual review.
- Runtime dependencies are pinned to the verified environment. Retraining with other package versions may require updating the dependency pins and saved model together.

## Troubleshooting

- **Model file missing:** run the notebook's tuning, tuned evaluation, and save cells; keep the generated file beside `app.py`.
- **`best_model` is not defined:** run the preceding notebook cells in order.
- **Package import error:** confirm the terminal and notebook use `.venv-sentiment`, then install `requirements.txt`.
- **Dataset not found:** launch the notebook from the project folder and check the CSV filename and location.
- **Model compatibility error:** retrain and save the model using the same environment used to run the app.

Load only Joblib model files you trust, because deserialization can execute code.