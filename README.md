## 📧 Spam Email Detection System using Machine Learning

A **production-grade machine learning system** designed to accurately classify emails as **Spam** or **Ham (legitimate)**.
The project follows a **modular pipeline architecture** for training and prediction and includes an **interactive Streamlit web application** for real-time usage.

---

## 🌐 Live Demo

🚀 **Try the deployed application here:**
👉 **[https://email-spam-detection-batch-b.streamlit.app/](https://email-spam-detection-batch-b.streamlit.app/)**

The web app supports:

* **Single Email Classification**
* **Batch `.mbox` Email Processing**
* Instant predictions using the trained ML model

---

## 📌 Project Overview

Email spam is a major cybersecurity and productivity concern. This project addresses the problem by building an automated spam detection system using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system is trained on a subset of the **Enron Email Dataset**, one of the most widely used real-world datasets for email classification research.

---

## 🎯 Objectives

* To preprocess and analyze real-world email data
* To build and evaluate multiple machine learning models
* To select the best-performing model using evaluation metrics
* To provide a user-friendly web interface for email classification
* To support both **single email prediction** and **bulk email (MBOX) processing**

---

## 🚀 Key Features

* ✅ Modular ML pipeline (Data Ingestion → Transformation → Training → Prediction)
* ✅ Multiple models (SVM, Logistic Regression, Decision Tree, KNN, Random Forest)
* ✅ TF-IDF based text feature extraction
* ✅ Automatic best model selection using F1-score
* ✅ Streamlit web app for real-time interaction
* ✅ Batch processing of `.mbox` email files
* ✅ Detailed logs and performance metrics

---

## 🛠️ Tech Stack

| Category             | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python 3.10+        |
| Frontend             | Streamlit           |
| Machine Learning     | Scikit-learn        |
| NLP                  | TF-IDF Vectorizer   |
| Data Handling        | Pandas, NumPy       |
| Dataset              | Enron Email Dataset |
| Version Control      | Git & GitHub        |


## 📂 Project Structure

```
Spam-Email-Detection/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
│
├── src/
│   ├── components/            # Data ingestion, transformation, training modules
│   ├── pipeline/              # Training & prediction pipelines
│   ├── config/                # Configuration files
│   └── utils/                 # Utility functions (logging, text cleaning)
│
├── data/
│   └── dataset/               # Email datasets (CSV / MBOX)
│
├── outputs/                   # Trained models & vectorizers
├── logs/                      # Execution logs
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Praneethvarma80022/email-spam-detection.git
cd Spam-Email-Detection
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.\.venv\Scripts\Activate.ps1
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

To train the machine learning models from scratch:

```bash
python -m src.pipeline.training_pipeline
```

This will:

* Train multiple ML models
* Perform cross-validation
* Select the best model
* Save the trained model and TF-IDF vectorizer in `outputs/`

---

## 🖥️ Running the Web Application

Start the Streamlit app:

```bash
streamlit run app.py
```

### Available Modes

* **Single Email Classification** – Paste email text and classify instantly
* **Batch MBOX Processing** – Upload `.mbox` file and download prediction results

---

## 📊 Model Evaluation Metrics

The system evaluates models using:

* Accuracy
* Precision
* Recall
* F1-Score

The best model is automatically selected based on **F1-Score**, which is ideal for imbalanced datasets like spam detection.

---

## 📁 Dataset Information

* **Dataset Used**: Enron Email Dataset
* **Description**: Email communications of ~150 employees from the Enron corporation
* **Usage**: A cleaned and preprocessed subset is used for training and testing

---

## 📌 Future Enhancements

* Deep Learning models (LSTM, BERT)
* Real-time email client integration
* Multilingual spam detection
* Deployment using Docker / Cloud platforms
* Model explainability (SHAP, LIME)

---

## 📜 License

This project is developed for **academic purposes** and is distributed under the **MIT License**.

---

## 👨‍🎓 Author

**Mudunuri Naga Kali Praneeth Varma**
Final Year B.Tech Project
Spam Email Detection using Machine Learning

