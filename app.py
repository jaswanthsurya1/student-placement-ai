import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from predict import predict_placement

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ibm_skillsbuild_placement_prediction_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==========================================
# Database Models
# ==========================================

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='student')  # 'student' or 'admin'
    predictions = db.relationship('Prediction', backref='student', lazy=True, cascade="all, delete-orphan")

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    register_number = db.Column(db.String(50), nullable=False, default='N/A')
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    cgpa = db.Column(db.Float, nullable=False)
    aptitude_score = db.Column(db.Integer, nullable=False)
    communication_skills = db.Column(db.Integer, nullable=False)
    programming_skills = db.Column(db.Integer, nullable=False)
    certifications = db.Column(db.Integer, nullable=False)
    internship_status = db.Column(db.String(10), nullable=False)  # 'Yes' or 'No'
    projects_completed = db.Column(db.Integer, nullable=False)
    prediction_result = db.Column(db.String(20), nullable=False)  # 'Placed' or 'Not Placed'
    placement_probability = db.Column(db.Float, nullable=False)   # %
    confidence_score = db.Column(db.Float, nullable=False)        # %
    suggested_role = db.Column(db.String(100), nullable=False)
    recommendations = db.Column(db.Text, nullable=False)          # Stored as JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# Routes & Controller Logic
# ==========================================

@app.route('/')
def home():
    """Home / Landing page."""
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles student account creation."""
    if 'user_id' in session:
        return redirect(url_for('dashboard' if session['user_role'] == 'student' else 'admin'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not name or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return render_template('register.html')
            
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template('register.html')
            
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')
            
        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists.", "danger")
            return render_template('register.html')
            
        # Create and save user
        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_pw, role='student')
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Please login below.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred during registration. Please try again.", "danger")
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticates both students and admins."""
    if 'user_id' in session:
        return redirect(url_for('dashboard' if session['user_role'] == 'student' else 'admin'))
        
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash("Both email and password are required.", "danger")
            return render_template('login.html')
            
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            flash(f"Welcome back, {user.name}!", "success")
            
            if user.role == 'admin':
                return redirect(url_for('admin'))
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password.", "danger")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Clears user session."""
    session.clear()
    flash("You have successfully logged out.", "info")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    """Renders student input dashboard."""
    if 'user_id' not in session or session['user_role'] != 'student':
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
        
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint triggered by student dashboard to perform AI inference."""
    if 'user_id' not in session or session['user_role'] != 'student':
        return jsonify({'error': 'Unauthorized access.'}), 401
        
    try:
        # Get request parameters (handles JSON and standard form POST)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
            
        # Extract fields
        student_name = data.get('student_name', '').strip()
        register_number = data.get('register_number', '').strip()
        age = int(data.get('age', 21))
        gender = data.get('gender', 'Male')
        department = data.get('department', 'CSE')
        cgpa = float(data.get('cgpa', 7.5))
        aptitude_score = int(data.get('aptitude_score', 70))
        communication_skills = int(data.get('communication_skills', 3))
        programming_skills = int(data.get('programming_skills', 3))
        certifications = int(data.get('certifications', 0))
        internship_status = data.get('internship_status', 'No')
        projects_completed = int(data.get('projects_completed', 1))
        
        # Validation checks
        if not student_name:
            return jsonify({'error': 'Student Name is required.'}), 400
        if not register_number:
            return jsonify({'error': 'Register / Roll Number is required.'}), 400
        if not (5.0 <= cgpa <= 10.0):
            return jsonify({'error': 'CGPA must be between 5.0 and 10.0.'}), 400
        if not (0 <= aptitude_score <= 100):
            return jsonify({'error': 'Aptitude Score must be between 0 and 100.'}), 400
        if not (1 <= communication_skills <= 5) or not (1 <= programming_skills <= 5):
            return jsonify({'error': 'Skill ratings must be between 1 and 5.'}), 400
        if not (0 <= certifications <= 10) or not (0 <= projects_completed <= 10):
            return jsonify({'error': 'Certifications/Projects must be positive and reasonable.'}), 400
            
        # Formulate inference dictionary
        # NOTE: Demographic variables (Age, Gender, Department) are intentionally
        # excluded from the input vector below to avoid algorithmic bias.
        inputs = {
            'cgpa': cgpa,
            'aptitude_score': aptitude_score,
            'communication_skills': communication_skills,
            'programming_skills': programming_skills,
            'internship_status': internship_status,
            'certifications': certifications,
            'projects_completed': projects_completed
        }
        
        # Run inference
        results = predict_placement(inputs)
        
        # Save to SQLite Database
        prediction_record = Prediction(
            user_id=session['user_id'],
            student_name=student_name,
            register_number=register_number,
            age=age,
            gender=gender,
            department=department,
            cgpa=cgpa,
            aptitude_score=aptitude_score,
            communication_skills=communication_skills,
            programming_skills=programming_skills,
            certifications=certifications,
            internship_status=internship_status,
            projects_completed=projects_completed,
            prediction_result=results['prediction'],
            placement_probability=results['probability'],
            confidence_score=results['confidence'],
            suggested_role=results['suggested_role'],
            recommendations=json.dumps(results['recommendations'])
        )
        
        db.session.add(prediction_record)
        db.session.commit()
        
        # Return response
        return jsonify({
            'success': True,
            'prediction': results['prediction'],
            'probability': results['probability'],
            'confidence': results['confidence'],
            'suggested_role': results['suggested_role'],
            'recommendations': results['recommendations']
        })
        
    except ValueError as val_err:
        return jsonify({'error': str(val_err)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f"Prediction failed due to server error: {e}"}), 500

@app.route('/history')
def history():
    """Renders prediction history for the authenticated student."""
    if 'user_id' not in session or session['user_role'] != 'student':
        flash("Please log in to view prediction history.", "warning")
        return redirect(url_for('login'))
        
    raw_predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    
    # Deserialize recommendations list from JSON string for each row
    predictions = []
    for pred in raw_predictions:
        pred_dict = {
            'id': pred.id,
            'student_name': pred.student_name,
            'register_number': pred.register_number,
            'department': pred.department,
            'cgpa': pred.cgpa,
            'aptitude_score': pred.aptitude_score,
            'communication_skills': pred.communication_skills,
            'programming_skills': pred.programming_skills,
            'certifications': pred.certifications,
            'internship_status': pred.internship_status,
            'projects_completed': pred.projects_completed,
            'prediction_result': pred.prediction_result,
            'placement_probability': pred.placement_probability,
            'confidence_score': pred.confidence_score,
            'suggested_role': pred.suggested_role,
            'recommendations': json.loads(pred.recommendations),
            'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M')
        }
        predictions.append(pred_dict)
        
    return render_template('history.html', predictions=predictions)

@app.route('/admin')
def admin():
    """Admin dashboard with high-level statistics and aggregations."""
    if 'user_id' not in session or session['user_role'] != 'admin':
        flash("Unauthorized. Administrator access only.", "danger")
        return redirect(url_for('login'))
        
    # Count of unique students by register number (active students)
    active_reg_nums = [r[0] for r in db.session.query(Prediction.register_number).distinct().all()]
    total_students = len(active_reg_nums)
    
    active_students_list = []
    for reg_num in active_reg_nums:
        latest_pred = Prediction.query.filter_by(register_number=reg_num).order_by(Prediction.timestamp.desc()).first()
        if latest_pred:
            active_students_list.append({
                'name': latest_pred.student_name,
                'register_number': latest_pred.register_number,
                'email': latest_pred.student.email if latest_pred.student else 'N/A',
                'department': latest_pred.department,
                'cgpa': latest_pred.cgpa,
                'latest_result': latest_pred.prediction_result,
                'latest_prob': latest_pred.placement_probability,
                'timestamp': latest_pred.timestamp.strftime('%Y-%m-%d %H:%M')
            })
                
    all_predictions = Prediction.query.all()
    total_predictions = len(all_predictions)
    
    # Calculate placement statistics
    placed_count = sum(1 for p in all_predictions if p.prediction_result == 'Placed')
    not_placed_count = total_predictions - placed_count
    
    placement_rate = round((placed_count / total_predictions * 100), 2) if total_predictions > 0 else 0.0
    
    # Calculate recommended skills distribution
    skill_counts = {}
    for p in all_predictions:
        try:
            recs = json.loads(p.recommendations)
            for r in recs:
                cat = r.get('category', 'General')
                skill_counts[cat] = skill_counts.get(cat, 0) + 1
        except Exception:
            pass
            
    # Sort recommendations by count descending
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get recent predictions
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    
    # Department-wise distribution
    dept_counts = {}
    for p in all_predictions:
        dept_counts[p.department] = dept_counts.get(p.department, 0) + 1
        
    # Gender distribution (for bias audits)
    gender_counts = {'Male': 0, 'Female': 0, 'Other': 0}
    for p in all_predictions:
        g = p.gender
        if g in gender_counts:
            gender_counts[g] += 1
        else:
            gender_counts['Other'] = gender_counts.get('Other', 0) + 1
            
    return render_template('admin.html',
                           total_students=total_students,
                           total_predictions=total_predictions,
                           placed_count=placed_count,
                           not_placed_count=not_placed_count,
                           placement_rate=placement_rate,
                           recommended_skills=sorted_skills[:5],
                           recent_predictions=recent_predictions,
                           active_students=active_students_list,
                           dept_counts_json=json.dumps(dept_counts),
                           gender_counts_json=json.dumps(gender_counts),
                           skill_counts_json=json.dumps(skill_counts))

@app.route('/performance')
def performance():
    """Displays accuracy and evaluation reports of the ML models."""
    metrics_file = "model/model_metrics.json"
    
    if not os.path.exists(metrics_file):
        flash("Model performance metrics have not been generated yet. Please run training.", "warning")
        return render_template('performance.html', metrics=None)
        
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
        
    return render_template('performance.html', metrics=metrics)

# ==========================================
# Application Context Initialization
# ==========================================

with app.app_context():
    # Setup database tables
    db.create_all()
    
    # Seed default Admin account if empty
    admin_exists = User.query.filter_by(role='admin').first()
    if not admin_exists:
        hashed_pw = generate_password_hash('AdminPassword123')
        default_admin = User(
            name="Placement Administrator",
            email="admin@placement.com",
            password=hashed_pw,
            role="admin"
        )
        db.session.add(default_admin)
        db.session.commit()
        print("Successfully seeded database with admin user (admin@placement.com / AdminPassword123).")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
