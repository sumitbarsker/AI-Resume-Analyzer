import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle

data = {
    "resume": [
        "Python machine learning tensorflow deep learning",
        "React html css javascript frontend ui",
        "Java spring boot backend mysql api",
        "Android kotlin firebase mobile app",
        "Data analysis pandas numpy sql power bi"
    ],

    "role": [
        "AI/ML Engineer",
        "Frontend Developer",
        "Backend Developer",
        "Android Developer",
        "Data Analyst"
    ]
}

df = pd.DataFrame(data)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", SVC())
])

model.fit(df["resume"], df["role"])

pickle.dump(
    model,
    open("model/resume_model.pkl", "wb")
)

print("Model trained successfully")
