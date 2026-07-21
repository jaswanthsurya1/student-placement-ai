import os
import numpy as np
import pandas as pd

def generate_placement_dataset(num_samples=1500, output_path="dataset/placement_data.csv"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    # 1. CGPA: normal-ish distribution, clipped to 5.0 - 10.0 range
    cgpa = np.random.normal(7.5, 1.0, num_samples)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    cgpa = np.round(cgpa, 2)
    
    # 2. Aptitude Score: 0 to 100, correlated slightly with CGPA
    aptitude_base = (cgpa / 10.0) * 80  # base score between 40 and 80
    aptitude_noise = np.random.normal(10, 10, num_samples)
    aptitude_score = np.clip(aptitude_base + aptitude_noise, 0, 100).astype(int)
    
    # 3. Communication Skills: 1 to 5 scale
    comm_skills = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.1, 0.2, 0.35, 0.25, 0.1])
    
    # 4. Programming Skills: 1 to 5 scale, correlated with CGPA
    prog_skills = []
    for c in cgpa:
        if c >= 9.0:
            p = [0.02, 0.08, 0.2, 0.4, 0.3]
        elif c >= 7.5:
            p = [0.05, 0.15, 0.4, 0.3, 0.1]
        elif c >= 6.0:
            p = [0.1, 0.4, 0.35, 0.1, 0.05]
        else:
            p = [0.4, 0.4, 0.15, 0.04, 0.01]
        prog_skills.append(np.random.choice([1, 2, 3, 4, 5], p=p))
    prog_skills = np.array(prog_skills)
    
    # 5. Internship Experience: Yes/No, correlated with CGPA & Prog Skills
    internship = []
    for c, p in zip(cgpa, prog_skills):
        prob = 0.1
        if c > 8.5 or p >= 4:
            prob = 0.65
        elif c > 7.0 or p >= 3:
            prob = 0.35
        internship.append(np.random.choice(['Yes', 'No'], p=[prob, 1 - prob]))
    internship = np.array(internship)
    
    # 6. Certifications: 0 to 4 counts
    certifications = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.2, 0.08, 0.02])
    
    # 7. Projects Completed: 0 to 4 counts
    projects = []
    for p in prog_skills:
        if p >= 4:
            p_dist = [0.05, 0.15, 0.4, 0.3, 0.1]
        elif p >= 3:
            p_dist = [0.15, 0.35, 0.3, 0.15, 0.05]
        else:
            p_dist = [0.4, 0.4, 0.15, 0.04, 0.01]
        projects.append(np.random.choice([0, 1, 2, 3, 4], p=p_dist))
    projects = np.array(projects)
    
    # Calculate Placement Likelihood Score
    # Convert Internship to 0/1 for score calculation
    internship_numeric = np.where(internship == 'Yes', 1.0, 0.0)
    
    # Normalize inputs for score weighting (0 to 1 scale)
    cgpa_norm = (cgpa - 5.0) / 5.0
    apt_norm = aptitude_score / 100.0
    comm_norm = (comm_skills - 1) / 4.0
    prog_norm = (prog_skills - 1) / 4.0
    cert_norm = certifications / 4.0
    proj_norm = projects / 4.0
    
    # Weight factors
    score = (
        0.30 * cgpa_norm +
        0.25 * apt_norm +
        0.15 * prog_norm +
        0.10 * comm_norm +
        0.08 * internship_numeric +
        0.07 * proj_norm +
        0.05 * cert_norm
    )
    
    # Add random variation to mimic real life
    noise = np.random.normal(0, 0.06, num_samples)
    final_score = score + noise
    
    # Determine placement status based on score threshold
    # Threshold at 0.45 represents ~55% placement rate
    placement_status = np.where(final_score >= 0.45, 'Placed', 'Not Placed')
    
    # Create DataFrame
    df = pd.DataFrame({
        'CGPA': cgpa,
        'AptitudeScore': aptitude_score,
        'CommunicationSkills': comm_skills,
        'ProgrammingSkills': prog_skills,
        'InternshipExperience': internship,
        'Certifications': certifications,
        'ProjectsCompleted': projects,
        'PlacementStatus': placement_status
    })
    
    # Let's introduce a very small amount of missing values (e.g. 1%) to demonstrate handling missing data
    # (Since the requirements state: "Perform Missing Value Handling")
    for col in ['CGPA', 'AptitudeScore', 'CommunicationSkills']:
        mask = np.random.rand(num_samples) < 0.015  # ~1.5% missingness
        df.loc[mask, col] = np.nan
        
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {num_samples} samples at: {output_path}")
    print(f"Placement Distribution:\n{df['PlacementStatus'].value_counts(normalize=True)}")

if __name__ == '__main__':
    generate_placement_dataset()
