import os
from predict import predict_placement

def run_tests():
    print("Testing ML Inference and Recommendation Module...")
    
    # Check if model files exist
    assert os.path.exists("model/best_model.joblib"), "best_model.joblib is missing!"
    assert os.path.exists("model/scaler.joblib"), "scaler.joblib is missing!"
    assert os.path.exists("model/model_metrics.json"), "model_metrics.json is missing!"
    print("Model files verified successfully.")
    
    # 1. Test case 1: High-performing student
    test_input_1 = {
        'cgpa': 9.2,
        'aptitude_score': 85,
        'communication_skills': 5,
        'programming_skills': 5,
        'internship_status': 'Yes',
        'certifications': 3,
        'projects_completed': 4
    }
    
    print("\nRunning Test Case 1 (High-performing student)...")
    res1 = predict_placement(test_input_1)
    print(f"Prediction: {res1['prediction']}")
    print(f"Placement Probability: {res1['probability']}%")
    print(f"Confidence Score: {res1['confidence']}%")
    print(f"Suggested Role: {res1['suggested_role']}")
    print(f"Number of recommendations generated: {len(res1['recommendations'])}")
    
    # Asserts
    assert res1['prediction'] == 'Placed', "Should be predicted as Placed"
    assert res1['probability'] > 50.0, "Probability should be > 50%"
    assert res1['suggested_role'] == "Software Development Engineer (SDE) / AI Engineer", "Should be suggested SDE"
    
    # 2. Test case 2: Low-performing student with specific gaps
    test_input_2 = {
        'cgpa': 5.8,
        'aptitude_score': 45,
        'communication_skills': 2,
        'programming_skills': 2,
        'internship_status': 'No',
        'certifications': 0,
        'projects_completed': 0
    }
    
    print("\nRunning Test Case 2 (Student with performance gaps)...")
    res2 = predict_placement(test_input_2)
    print(f"Prediction: {res2['prediction']}")
    print(f"Placement Probability: {res2['probability']}%")
    print(f"Confidence Score: {res2['confidence']}%")
    print(f"Suggested Role: {res2['suggested_role']}")
    print(f"Number of recommendations generated: {len(res2['recommendations'])}")
    
    # Verify recommendations contain specific actions
    categories = [r['category'] for r in res2['recommendations']]
    print("Generated Recommendation Categories:", categories)
    
    assert res2['prediction'] == 'Not Placed', "Should be predicted as Not Placed"
    assert 'Aptitude Practice' in categories, "Aptitude recommendation should trigger"
    assert 'Communication Training' in categories, "Communication recommendation should trigger"
    assert 'Data Structures & Programming' in categories, "Programming recommendation should trigger"
    assert 'Practical Internship' in categories, "Internship recommendation should trigger"
    assert 'Project Development' in categories, "Project recommendation should trigger"
    
    print("\nAll programmatic tests passed successfully! The prediction pipeline is ready.")

if __name__ == '__main__':
    run_tests()
