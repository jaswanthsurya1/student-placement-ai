import os
import json
import random
from app import app, db, User, Prediction
from werkzeug.security import generate_password_hash
from predict import predict_placement

def seed_database():
    print("Seeding database with realistic student data...")
    
    with app.app_context():
        # Recreate tables
        db.drop_all()
        db.create_all()
        
        # 1. Seed Admin User
        admin_hashed = generate_password_hash("AdminPassword123")
        admin = User(
            name="Placement Administrator",
            email="admin@placement.com",
            password=admin_hashed,
            role="admin"
        )
        db.session.add(admin)
        
        # 2. Seed Student Users
        students_info = [
            ("Amit Sharma", "amit@placement.com", "CSE", "Male"),
            ("Priya Patel", "priya@placement.com", "IT", "Female"),
            ("Rahul Verma", "rahul@placement.com", "ECE", "Male"),
            ("Sneha Reddy", "sneha@placement.com", "CSE", "Female"),
            ("Vikram Singh", "vikram@placement.com", "ME", "Male"),
            ("Ananya Das", "ananya@placement.com", "IT", "Female"),
            ("Rohan Mehta", "rohan@placement.com", "ECE", "Male"),
            ("Kavita Nair", "kavita@placement.com", "EEE", "Female"),
            ("Siddharth Sen", "siddharth@placement.com", "ME", "Male"),
            ("Neha Gupta", "neha@placement.com", "CE", "Female")
        ]
        
        student_users = []
        for name, email, dept, gender in students_info:
            hashed_pw = generate_password_hash("StudentPassword123")
            student = User(
                name=name,
                email=email,
                password=hashed_pw,
                role="student"
            )
            db.session.add(student)
            student_users.append((student, dept, gender))
            
        # Commit users to generate IDs
        db.session.commit()
        print(f"Created {len(student_users)} student accounts and 1 admin account.")
        
        # 3. Seed Predictions for Students
        # We will create 15 predictions with varying skill profiles to populate charts
        departments = ["CSE", "ECE", "EEE", "ME", "CE", "IT"]
        
        profiles = [
            # High profile CSE
            {"cgpa": 9.1, "aptitude": 88, "prog": 5, "comm": 4, "cert": 2, "intern": "Yes", "proj": 3},
            # High profile IT
            {"cgpa": 8.7, "aptitude": 82, "prog": 4, "comm": 4, "cert": 1, "intern": "Yes", "proj": 2},
            # Medium profile ECE
            {"cgpa": 7.8, "aptitude": 72, "prog": 3, "comm": 3, "cert": 0, "intern": "No", "proj": 2},
            # Medium profile CSE (placed due to prog & projects)
            {"cgpa": 7.2, "aptitude": 75, "prog": 4, "comm": 3, "cert": 1, "intern": "Yes", "proj": 3},
            # Low profile ME (not placed)
            {"cgpa": 6.1, "aptitude": 55, "prog": 2, "comm": 2, "cert": 0, "intern": "No", "proj": 1},
            # Low profile CE (not placed)
            {"cgpa": 5.9, "aptitude": 48, "prog": 1, "comm": 3, "cert": 0, "intern": "No", "proj": 0},
            # High profile EEE
            {"cgpa": 8.4, "aptitude": 80, "prog": 3, "comm": 5, "cert": 3, "intern": "No", "proj": 2},
            # Medium profile IT (placed)
            {"cgpa": 7.9, "aptitude": 78, "prog": 4, "comm": 4, "cert": 2, "intern": "Yes", "proj": 2},
            # Borderline profile CSE (placed)
            {"cgpa": 7.4, "aptitude": 68, "prog": 3, "comm": 4, "cert": 1, "intern": "No", "proj": 2},
            # Borderline profile ECE (not placed)
            {"cgpa": 6.8, "aptitude": 62, "prog": 2, "comm": 3, "cert": 0, "intern": "No", "proj": 1},
            # Another high profile ME
            {"cgpa": 8.6, "aptitude": 85, "prog": 3, "comm": 4, "cert": 2, "intern": "Yes", "proj": 2},
            # Very low EEE (not placed)
            {"cgpa": 5.4, "aptitude": 40, "prog": 2, "comm": 2, "cert": 0, "intern": "No", "proj": 0},
            # High profile CE (placed)
            {"cgpa": 8.1, "aptitude": 76, "prog": 3, "comm": 4, "cert": 2, "intern": "No", "proj": 2},
            # Medium profile ME (not placed)
            {"cgpa": 6.9, "aptitude": 65, "prog": 2, "comm": 3, "cert": 1, "intern": "No", "proj": 1},
            # High profile IT (placed)
            {"cgpa": 8.9, "aptitude": 90, "prog": 5, "comm": 5, "cert": 3, "intern": "Yes", "proj": 4}
        ]
        
        for i, profile in enumerate(profiles):
            # Select student cyclically
            student, dept, gender = student_users[i % len(student_users)]
            
            inputs = {
                'cgpa': profile["cgpa"],
                'aptitude_score': profile["aptitude"],
                'communication_skills': profile["comm"],
                'programming_skills': profile["prog"],
                'internship_status': profile["intern"],
                'certifications': profile["cert"],
                'projects_completed': profile["proj"]
            }
            
            # Predict
            res = predict_placement(inputs)
            
            # Save Prediction record
            pred_record = Prediction(
                user_id=student.id,
                student_name=student.name,
                age=random.choice([20, 21, 22]),
                gender=gender,
                department=dept,
                cgpa=profile["cgpa"],
                aptitude_score=profile["aptitude"],
                communication_skills=profile["comm"],
                programming_skills=profile["prog"],
                certifications=profile["cert"],
                internship_status=profile["intern"],
                projects_completed=profile["proj"],
                prediction_result=res['prediction'],
                placement_probability=res['probability'],
                confidence_score=res['confidence'],
                suggested_role=res['suggested_role'],
                recommendations=json.dumps(res['recommendations'])
            )
            db.session.add(pred_record)
            
        db.session.commit()
        print(f"Successfully seeded {len(profiles)} placement prediction records.")

if __name__ == '__main__':
    seed_database()
