import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate_models(data_path="dataset/placement_data.csv", model_dir="model"):
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # 1. Data Cleaning: Missing Value Handling
    # Separate numeric and categorical imputations
    num_cols_with_nan = ['CGPA', 'AptitudeScore']
    cat_cols_with_nan = ['CommunicationSkills']
    
    # Impute continuous values with median
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols_with_nan] = num_imputer.fit_transform(df[num_cols_with_nan])
    
    # Impute categorical/ordinal skill level with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols_with_nan] = cat_imputer.fit_transform(df[cat_cols_with_nan])
    
    # Convert CommunicationSkills back to integer
    df['CommunicationSkills'] = df['CommunicationSkills'].astype(int)
    
    # 2. Label Encoding for categorical features
    # InternshipExperience: 'Yes' -> 1, 'No' -> 0
    le_intern = LabelEncoder()
    df['InternshipExperience'] = le_intern.fit_transform(df['InternshipExperience'].astype(str))
    # Note: 'No' is encoded as 0, 'Yes' is encoded as 1.
    
    # PlacementStatus: 'Placed' -> 1, 'Not Placed' -> 0
    le_placement = LabelEncoder()
    df['PlacementStatus'] = le_placement.fit_transform(df['PlacementStatus'].astype(str))
    # Note: 'Not Placed' is 0, 'Placed' is 1.
    
    # Define features and target
    feature_cols = [
        'CGPA', 
        'AptitudeScore', 
        'CommunicationSkills', 
        'ProgrammingSkills', 
        'InternshipExperience', 
        'Certifications', 
        'ProjectsCompleted'
    ]
    X = df[feature_cols]
    y = df['PlacementStatus']
    
    # 3. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and label encoders for predictions
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le_intern, os.path.join(model_dir, 'le_intern.joblib'))
    joblib.dump(le_placement, os.path.join(model_dir, 'le_placement.joblib'))
    
    # 5. Define Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
    }
    
    metrics_summary = {}
    best_accuracy = 0
    best_model_name = None
    best_model_obj = None
    
    # 6. Train & Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        metrics_summary[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        print(f"{name} Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Track the best model based on accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # 7. Save the best model
    best_model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(best_model_obj, best_model_path)
    print(f"Saved best model ({best_model_name}) to: {best_model_path}")
    
    # Save the metrics to display on the frontend performance page
    performance_metadata = {
        'best_model': best_model_name,
        'features': feature_cols,
        'metrics': metrics_summary
    }
    
    metrics_path = os.path.join(model_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(performance_metadata, f, indent=4)
    print(f"Saved performance metrics to: {metrics_path}")

if __name__ == '__main__':
    train_and_evaluate_models()
