import pandas as pd
from pathlib import Path
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def lr_words(df, clean = True):

    if clean:
        nlp = spacy.load("en_core_web_sm")
        texts = df["text"].astype(str).str.lower().tolist()
        clean_texts = []

        for doc in nlp.pipe(texts, batch_size=1000, n_process = 4):
            clean_texts.append(
                " ".join(
                    token.lemma_
                    for token in doc
                    if not token.is_stop and token.is_alpha
                )
            )

        df["clean_text"] = clean_texts
    
    if "clean_text" in df.columns:
        x = df["clean_text"]
    else:
        x = df["text"].astype(str).str.lower()
    y = df["sentiment"]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    vectorizer = CountVectorizer(min_df = 3)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=500, solver="liblinear", class_weight="balanced"
    )
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)

    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred), model


if __name__ == "__main__":
    cwd = Path.cwd()
    data_path = cwd / "logistic_regression" / "data" / "clean_amazon_reviews.csv"
    
    data = pd.read_csv(data_path, encoding="utf-8")
    
    df = data[["sentiment", "title", "text", "clean_text"]].copy()

    df["sentiment"] = df["sentiment"].replace(1, -1)
    df["sentiment"] = df["sentiment"].replace(2, 1)
    
    acc, report, model = lr_words(df, clean = False)
    print("Accuracy", acc)
    print("Classification Report")
    print(report)