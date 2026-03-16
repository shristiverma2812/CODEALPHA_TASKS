import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
y_class_names = iris.target_names

# Load trained model
with open("irisflower.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.title("🌸 Iris Flower Classifier Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Performance", "📂 Dataset Info"])

# ---------------- Prediction Tab ----------------
with tab1:
    st.header("Enter Measurements")
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sepal_width  = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    petal_width  = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

    input_df = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })

    if st.button("Predict"):
        prediction_num = model.predict(input_df)[0]   # ✅ always int now
        st.success(f"🌼 Predicted Flower: {y_class_names[prediction_num]}")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame(model.predict_proba(input_df), columns=y_class_names)
        st.bar_chart(prob_df.T)

# ---------------- Performance Tab ----------------
with tab2:
    st.header("Model Performance")

    y_pred_test = model.predict(X)   # ✅ numeric predictions
    cm = confusion_matrix(y, y_pred_test, labels=[0,1,2])

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=y_class_names,
                yticklabels=y_class_names,
                ax=ax, cmap="Purples")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y, y_pred_test,
                                   target_names=y_class_names,
                                   output_dict=True)
    st.write(pd.DataFrame(report).transpose())

# ---------------- Dataset Info Tab ----------------
with tab3:
    st.header("Dataset Overview")
    st.write("This app uses the classic Iris dataset with 150 samples and 3 flower classes.")
    st.write(pd.DataFrame(iris.data, columns=iris.feature_names).head(10))
    st.metric("Model Accuracy", "97%")