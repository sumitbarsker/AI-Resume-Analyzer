import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle

# Sample dataset
data = {
    "resume": [
        "Python machine learning deep learning tensorflow",
        "React javascript html css frontend developer",
        "Java spring boot backend api mysql",
        "Android kotlin firebase xml mobile app",
        "Data analysis pandas numpy power bi sql"
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

# ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", SVC())
])

# Train model
model.fit(df["resume"], df["role"])

# Save model
pickle.dump(model, open("model/resume_model.pkl", "wb"))

print("Model trained successfully")