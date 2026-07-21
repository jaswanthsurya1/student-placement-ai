import os
import numpy as np
import pandas as pd
import joblib

def load_prediction_artifacts(model_dir="model"):
    """Loads all model artifacts necessary for prediction."""
    try:
        model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        le_intern = joblib.load(os.path.join(model_dir, 'le_intern.joblib'))
        return model, scaler, le_intern
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None

def generate_recommendations(inputs):
    """Generates specific skill recommendations based on student inputs."""
    recs = []
    
    # 1. Aptitude Score
    if inputs['aptitude_score'] < 70:
        recs.append({
            'category': 'Aptitude Practice',
            'severity': 'High' if inputs['aptitude_score'] < 50 else 'Medium',
            'message': f"Your aptitude score of {inputs['aptitude_score']}% is below the target (70%+). Practice quantitative aptitude, logical reasoning, and verbal ability on platforms like IndiaBIX, FacePrep, or GeeksforGeeks."
        })
        
    # 2. Communication Skills
    if inputs['communication_skills'] <= 3:
        severity = 'High' if inputs['communication_skills'] <= 2 else 'Medium'
        recs.append({
            'category': 'Communication Training',
            'severity': severity,
            'message': f"Communication skill level ({inputs['communication_skills']}/5) needs enhancement. Practice active listening, participate in mock group discussions (GDs), and consider public speaking platforms like Toastmasters or online communication workshops."
        })
        
    # 3. Programming Skills
    if inputs['programming_skills'] <= 3:
        severity = 'High' if inputs['programming_skills'] <= 2 else 'Medium'
        recs.append({
            'category': 'Data Structures & Programming',
            'severity': severity,
            'message': f"Programming skills rating ({inputs['programming_skills']}/5) should be improved. Focus on learning core Data Structures and Algorithms (DSA) in Python, Java, or C++, and solve at least 1-2 coding problems daily on LeetCode, HackerRank, or GeeksforGeeks."
        })
        
    # 4. Internship Experience
    if inputs['internship_status'].lower() == 'no':
        recs.append({
            'category': 'Practical Internship',
            'severity': 'Medium',
            'message': "Having no prior internship experience reduces placement chances. Apply for virtual internships (e.g., IBM SkillsBuild, Forage) or target open-source contributions and local startup internships to gain real-world industry experience."
        })
        
    # 5. Certifications
    if inputs['certifications'] == 0:
        recs.append({
            'category': 'Professional Certification',
            'severity': 'Medium',
            'message': "Boost your resume! You have 0 certifications. You can gain industry-recognized certifications directly on our platform through IBM SkillsBuild. <a href='https://skillsbuild.org/' target='_blank' class='fw-bold text-primary text-decoration-none'><i class='bi bi-box-arrow-up-right'></i> Explore Certification Courses</a>."
        })
        
    # 6. Projects Completed
    if inputs['projects_completed'] <= 1:
        severity = 'High' if inputs['projects_completed'] == 0 else 'Medium'
        recs.append({
            'category': 'Project Development',
            'severity': severity,
            'message': f"You have completed only {inputs['projects_completed']} projects. Build at least 2 robust, end-to-end projects (e.g., full-stack web applications, machine learning dashboards, or mobile apps), host their code on GitHub, and deploy them on Render or Vercel."
        })
        
    # Default recommendation if the student is already outstanding
    if not recs:
        recs.append({
            'category': 'Interview Preparation',
            'severity': 'Low',
            'message': "Excellent profile! Keep practicing advanced mock technical interviews, refine your resume, and work on behavioral questions (using the STAR method) to ace premium product-based company placements."
        })
        
    return recs

def recommend_job_role(inputs):
    """Recommends a suitable job role based on academic performance and skills."""
    cgpa = inputs['cgpa']
    prog = inputs['programming_skills']
    comm = inputs['communication_skills']
    apt = inputs['aptitude_score']
    
    if prog >= 4 and cgpa >= 8.0:
        if apt >= 80:
            return "Software Development Engineer (SDE) / AI Engineer"
        else:
            return "Full Stack Developer"
    elif prog >= 4:
        return "Backend Developer / QA Automation Engineer"
    elif prog == 3 and cgpa >= 7.5:
        return "Associate Software Engineer / Systems Analyst"
    elif comm >= 4 and prog <= 3:
        if apt >= 75:
            return "Business Analyst / Technology Consultant"
        else:
            return "Product Support Specialist / Tech Sales Specialist"
    elif prog >= 3:
        return "Frontend Developer / Application Engineer"
    else:
        return "Technical Support Engineer / QA Manual Tester"

def predict_placement(inputs, model_dir="model"):
    """
    Predicts student placement and returns complete results.
    inputs dict structure:
    {
        'cgpa': float,
        'aptitude_score': int,
        'communication_skills': int,
        'programming_skills': int,
        'internship_status': 'Yes'/'No',
        'certifications': int,
        'projects_completed': int
    }
    """
    model, scaler, le_intern = load_prediction_artifacts(model_dir)
    
    if model is None or scaler is None or le_intern is None:
        raise ValueError("Model artifacts are missing or failed to load. Ensure you have run train_model.py first.")
        
    # Preprocess inputs
    # Encode internship: 'Yes' -> 1, 'No' -> 0 using label encoder, or fall back to mapping
    try:
        internship_encoded = le_intern.transform([inputs['internship_status']])[0]
    except Exception:
        internship_encoded = 1 if inputs['internship_status'].lower() == 'yes' else 0
        
    # Construct feature array in exact training order:
    # ['CGPA', 'AptitudeScore', 'CommunicationSkills', 'ProgrammingSkills', 'InternshipExperience', 'Certifications', 'ProjectsCompleted']
    feature_values = [
        inputs['cgpa'],
        inputs['aptitude_score'],
        inputs['communication_skills'],
        inputs['programming_skills'],
        internship_encoded,
        inputs['certifications'],
        inputs['projects_completed']
    ]
    
    # Scale features
    features_df = pd.DataFrame([feature_values], columns=[
        'CGPA', 'AptitudeScore', 'CommunicationSkills', 'ProgrammingSkills', 'InternshipExperience', 'Certifications', 'ProjectsCompleted'
    ])
    features_scaled = scaler.transform(features_df)
    
    # Run prediction
    prediction_class = model.predict(features_scaled)[0]
    
    # Calculate probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        # probability of Placement (class 1)
        placed_probability = float(probabilities[1])
    else:
        # Fallback for models without predict_proba (should not happen with our selection)
        placed_probability = 1.0 if prediction_class == 1 else 0.0
        
    # Confidence Score: probability of the predicted outcome
    if prediction_class == 1:
        prediction_result = "Placed"
        confidence_score = placed_probability
    else:
        prediction_result = "Not Placed"
        confidence_score = 1.0 - placed_probability
        
    # Generate recommendations and role
    recommendations = generate_recommendations(inputs)
    suggested_role = recommend_job_role(inputs)
    
    return {
        'prediction': prediction_result,
        'probability': round(placed_probability * 100, 2),
        'confidence': round(confidence_score * 100, 2),
        'suggested_role': suggested_role,
        'recommendations': recommendations
    }
