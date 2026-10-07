import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# Load Dataset and Model
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/student_performance_cleaned.csv"
    )


@st.cache_resource
def load_model():
    return joblib.load(
        "models/final_marks_model.joblib"
    )


df = load_data()
model = load_model()


# --------------------------------------------------
# Performance Category Function
# --------------------------------------------------

def get_category(marks):

    if marks >= 90:
        return "Excellent"

    elif marks >= 75:
        return "Very Good"

    elif marks >= 60:
        return "Good"

    elif marks >= 50:
        return "Average"

    else:
        return "Needs Improvement"


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("🎓 Student Performance")

page = st.sidebar.radio(
    "Navigate to",
    ["Dashboard", "Prediction"]
)


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.title("🎓 Student Performance Analysis")
    st.subheader("Dashboard")

    st.write(
        "Overview of the cleaned student performance dataset."
    )

    # Calculate dashboard metrics
    total_students = len(df)

    average_marks = df["Final_Exam_Marks"].mean()

    average_attendance = df["Attendance_Percentage"].mean()

    pass_percentage = (
        (df["Pass_Fail"] == "Pass").mean() * 100
    )

    highest_marks = df["Final_Exam_Marks"].max()

    lowest_marks = df["Final_Exam_Marks"].min()

    # Display metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Students",
        total_students
    )

    col2.metric(
        "Average Final Marks",
        f"{average_marks:.2f}"
    )

    col3.metric(
        "Average Attendance",
        f"{average_attendance:.2f}%"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Pass Percentage",
        f"{pass_percentage:.2f}%"
    )

    col5.metric(
        "Highest Marks",
        f"{highest_marks:.0f}"
    )

    col6.metric(
        "Lowest Marks",
        f"{lowest_marks:.0f}"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Prediction":

    st.title("📊 Student Performance Prediction")

    st.write(
        "Enter the student's academic information "
        "to estimate the final exam marks."
    )

    st.info(
        "This prediction uses five core academic features."
    )

    # ----------------------------------------------
    # Input Fields
    # ----------------------------------------------

    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.5
    )

    previous_marks = st.number_input(
        "Previous Semester Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=0.5
    )

    assignment = st.number_input(
        "Assignment Average",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5
    )

    internal_marks = st.number_input(
        "Internal Test Marks",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.5
    )

    # ----------------------------------------------
    # Prediction Button
    # ----------------------------------------------

    if st.button(
        "Predict Final Marks",
        type="primary"
    ):

        try:

            # Create input DataFrame
            input_data = pd.DataFrame({
                "Study_Hours_Per_Day": [study_hours],
                "Attendance_Percentage": [attendance],
                "Previous_Semester_Marks": [previous_marks],
                "Assignment_Average": [assignment],
                "Internal_Test_Marks": [internal_marks]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Keep prediction between 0 and 100
            prediction = max(
                0,
                min(100, prediction)
            )

            # Get performance category
            category = get_category(prediction)

            st.success("Prediction completed successfully!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Final Marks",
                f"{prediction:.2f} / 100"
            )

            col2.metric(
                "Performance Category",
                category
            )

        except Exception as e:

            st.error(
                f"Unable to make prediction: {e}"
            )

    # ----------------------------------------------
    # Disclaimer
    # ----------------------------------------------

    st.warning(
        "This is an estimate from a practice model and "
        "is not an official academic decision."
    )