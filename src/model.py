import os
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Paths
DATA_PATH = "data/processed/student_performance_cleaned.csv"
MODEL_PATH = "models/final_marks_model.joblib"


# Five core features required by the project
FEATURES = [
    "Study_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Semester_Marks",
    "Assignment_Average",
    "Internal_Test_Marks"
]

TARGET = "Final_Exam_Marks"


def train_model():

    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # Select features and target
    X = df[FEATURES]
    y = df[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Create pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Create models directory if needed
    os.makedirs("models", exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()