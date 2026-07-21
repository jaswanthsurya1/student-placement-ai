# AI-Powered Student Placement Prediction & Skill Recommendation System

An end-to-end Web Application and Machine Learning solution designed for final-year college students to evaluate their placement readiness, predict recruitment outcomes, and receive tailored skill recommendations to upgrade their profiles.

This project is structured as a professional, beginner-friendly submission suitable for an **IBM SkillsBuild Internship**.

---

## 🚀 Key Features

1. **AI Placement Predictor**: Leverages machine learning models (Logistic Regression, Decision Tree, Random Forest, KNN) to predict placement status and estimate probability.
2. **Dynamic Skill Recommendations**: A rules-engine checks student deficiencies (low CGPA, low aptitude, lack of projects, no internships) and supplies direct, actionable steps for improvement.
3. **Seeded Admin Dashboard**: Visualizes placement outcomes, department distributions, gender representation audits, and skill deficiency charts via **Chart.js**.
4. **Transparent Model Performance**: A dedicated page comparing the evaluation metrics (Accuracy, Precision, Recall, F1 Score) of all trained algorithms alongside the Confusion Matrix of the active model.
5. **Ethics & Fairness (Demographic Bias Mitigation)**: Personal details (Age, Gender, Department) are collected for tracking and student record preservation but **intentionally excluded** from the ML feature vector to prevent demographic discrimination.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (IBM Blue custom styling), Bootstrap 5, Chart.js
- **Backend**: Python 3, Flask, Flask-SQLAlchemy (SQLite)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib

---

## 📂 Project Structure

```text
project/
│
├── app.py                  # Core Flask server with routes, auth, and DB models
├── predict.py              # Inferences execution, role selection, and skill recommendation logic
├── train_model.py          # Cleans data, trains, compares, and serializes the best ML model
├── generate_dataset.py     # Generates synthetic data with realistic correlations for training
├── seed_db.py              # Seeds the SQLite database with dummy students and histories
├── test_prediction.py      # Run automated verification tests on prediction pipelines
├── database.db             # SQLite Database (created on first run / seeded)
├── requirements.txt        # Specifies all project dependencies
├── README.md               # Complete project documentation and guide
│
├── model/                  # Serialized ML artifacts
│   ├── best_model.joblib   # Trained Scikit-Learn classifier
│   ├── scaler.joblib       # Fitted StandardScaler
│   ├── le_intern.joblib    # Label encoder for internships
│   ├── le_placement.joblib # Label encoder for placement status
│   └── model_metrics.json  # Stored evaluations and comparisons
│
├── dataset/
│   └── placement_data.csv  # Generated CSV dataset containing student profiles
│
└── static/
    ├── css/
    │   └── style.css       # Custom premium IBM Blue stylesheet
    └── js/                 # Folder for client scripts if separated
```

---

## 📊 Database Schema

### 1. `users` Table (Authentication)
- `id` (INTEGER, Primary Key): Unique Identifier.
- `name` (TEXT, Not Null): User's Full Name.
- `email` (TEXT, Unique, Not Null): Account Email.
- `password` (TEXT, Not Null): Hashed password.
- `role` (TEXT, Default 'student'): Roles are 'student' or 'admin'.

### 2. `predictions` Table (History & Logs)
- `id` (INTEGER, Primary Key).
- `user_id` (INTEGER, Foreign Key): Links prediction to the student.
- `student_name` (TEXT): Display name for the record.
- `age` (INTEGER), `gender` (TEXT), `department` (TEXT): Demographic data (used for audits/history, **not** input to ML).
- `cgpa` (REAL), `aptitude_score` (INTEGER), `communication_skills` (INTEGER), `programming_skills` (INTEGER), `certifications` (INTEGER), `internship_status` (TEXT), `projects_completed` (INTEGER): Merit/skill features.
- `prediction_result` (TEXT): Predicted class ('Placed' or 'Not Placed').
- `placement_probability` (REAL): Likelihood of class 'Placed' (0% to 100%).
- `confidence_score` (REAL): Prediction confidence (probability of predicted class).
- `suggested_role` (TEXT): Selected job profile (e.g. SDE, Business Analyst).
- `recommendations` (TEXT): JSON string of actionable study steps.
- `timestamp` (DATETIME): Time of execution.

---

## 🤖 Machine Learning Pipeline

1. **Synthetic Data Generation** (`generate_dataset.py`): Creates `placement_data.csv` with 1500 students. Correlates CGPA, Aptitude, Projects, and Internships with Placement Status. Simulates 1.5% missing value noise.
2. **Missing Value Handling**: In `train_model.py`, continuous columns (CGPA, Aptitude) are imputed with their **medians**. Ordinal/categorical fields (Communication) are imputed with their **mode** (most frequent).
3. **Encoding & Scaling**:
   - Categorical values like `InternshipExperience` ('Yes'/'No') are encoded to $1/0$.
   - Features are normalized using `StandardScaler` to ensure scale differences (e.g. CGPA on 10.0 scale vs Aptitude on 100% scale) do not skew model weights.
4. **Model Comparison & Selection**:
   - Logistic Regression, Decision Tree, Random Forest, and KNN are trained on an 80/20 train/test split.
   - The script automatically saves the model with the highest test accuracy.
   - Saves metrics to `model_metrics.json`.

---

## ⚙️ Installation & Usage Guide

### Step 1: Clone or Open Project Folder
Open your terminal and navigate to the project directory:
```bash
cd "d:\ibm project"
```

### Step 2: Install Dependencies
Run the package manager to install core requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Train Models & Generate Artifacts
Run the dataset generator, then execute training:
```bash
python generate_dataset.py
python train_model.py
```
*This produces `best_model.joblib` and `model_metrics.json` inside the `model/` folder.*

### Step 4: Programmatic Verification
Verify the ML prediction pipeline and check rules-engine recommendations:
```bash
python test_prediction.py
```

### Step 5: Seed the SQLite Database
Prepopulate registration and predictions tables to enable graphs:
```bash
python seed_db.py
```

### Step 6: Start the Web Application
Launch the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
**`http://127.0.0.1:5000/`**

---

## 🔑 Login Accounts for Review

### 1. Student Account
- **Email**: `amit@placement.com` (or any seeded student)
- **Password**: `StudentPassword123`
*(Allows student inputs, AI predictions, and displays custom history logs)*

### 2. Admin Account
- **Email**: `admin@placement.com`
- **Password**: `AdminPassword123`
*(Allows viewing student numbers, prediction trends, placement rate, gender counts, and top skill gaps)*
