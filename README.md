# Fake News Detection Using Machine Learning

## Project Overview

This project is a Machine Learning-based web application that predicts whether a news article is **Real** or **Fake**.

The model uses **TF-IDF Vectorization** and a **Passive Aggressive Classifier** for text classification. The application is deployed using **Streamlit Community Cloud**.

## Features

* Predicts whether a news article is Real or Fake.
* Simple and user-friendly Streamlit interface.
* Fast prediction using a trained Machine Learning model.
* Reset button to clear the input text.
* Deployed as a live web application.

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Joblib
* Streamlit

## Machine Learning Algorithm

* TF-IDF Vectorizer
* Passive Aggressive Classifier

## Model Performance

* Accuracy: **99%**

## Project Structure

```text
Fake-News-Detection/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── dataset/
│   ├── Fake.csv
│   └── True.csv
└── model/
    ├── fake_news_model.pkl
    └── vectorizer.pkl
```

## Live Demo

Streamlit App:
https://fake-news-detection-5buhwtdjweu5spjrpdh5dc.streamlit.app/

## How to Run Locally

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Author

**Akash Kumar Mandal**

Data Science Internship Project
Pinnacle Labs
2026
