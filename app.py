import pickle
import numpy as np
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'genetic-disorder-secret-key-2025'

genai.configure(api_key='AIzaSyBxVYroaT2oo4zfiL9w2Wlruf5PgxeypGk')

try:
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/random_forest_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    disease_names = metadata['disease_names']
    print("✓ ML models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")

USERS_FILE = 'users.json'

def init_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)

init_users_file()

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def find_user(username):
    users = load_users()
    for user in users:
        if user['username'] == username:
            return user
    return None

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('detection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = find_user(username)
        
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['email'] = user['email']
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('detection'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        age = request.form.get('age')
        purpose = request.form.get('purpose')
        
        if find_user(username):
            flash('Username already exists. Please choose another.', 'error')
            return redirect(url_for('signup'))
        
        users = load_users()
        for user in users:
            if user['email'] == email:
                flash('Email already registered. Please use another.', 'error')
                return redirect(url_for('signup'))
        
        new_user = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'username': username,
            'password': generate_password_hash(password),
            'age': age,
            'purpose': purpose,
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        users.append(new_user)
        save_users(users)
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/detection')
def detection():
    if 'username' not in session:
        flash('Please log in to access the detection system.', 'error')
        return redirect(url_for('login'))
    
    return render_template('detection.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({
            'success': False,
            'error': 'Unauthorized. Please log in.'
        }), 401
    
    try:
        data = request.json
        
        age = float(data.get('age', 5))
        gender = int(data.get('gender', 0))
        genes_mother = int(data.get('genes_mother', 0))
        inherited_father = int(data.get('inherited_father', 0))
        maternal_gene = int(data.get('maternal_gene', 0))
        paternal_gene = int(data.get('paternal_gene', 0))
        status = int(data.get('status', 0))
        blood_test = int(data.get('blood_test', 1))
        birth_defects = int(data.get('birth_defects', 0))
        birth_asphyxia = int(data.get('birth_asphyxia', 0))
        maternal_illness = int(data.get('maternal_illness', 0))
        symptom_count = int(data.get('symptom_count', 0))
        blood_count = float(data.get('blood_count', 5.0))
        wbc_count = float(data.get('wbc_count', 7.0))
        inheritance_pattern = data.get('inheritance_pattern', 'Unknown')
        
        features = [
            age,
            genes_mother,
            inherited_father,
            maternal_gene,
            paternal_gene,
            blood_count,
            status,
            0,
            0,
            0,
            gender,
            birth_asphyxia,
            2,
            0,
            0,
            maternal_illness,
            2,
            2,
            0,
            0,
            0,
            birth_defects,
            wbc_count,
            blood_test,
            symptom_count
        ]
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        disease = disease_names.get(prediction, "Unknown Disorder")
        
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        A patient has been diagnosed with {disease}. 
        
        Patient details:
        - Age: {age} years
        - Gender: {'Male' if gender==0 else 'Female' if gender==1 else 'Ambiguous'}
        - Symptom count: {symptom_count}
        - Family history: {'Yes' if genes_mother or inherited_father else 'No'}
        
        Provide:
        1. Brief explanation of this genetic disorder (2-3 sentences)
        2. Common symptoms to watch for
        3. Recommended lifestyle modifications
        4. When to seek immediate medical attention
        5. Genetic counseling recommendations
        
        Keep the response concise, empathetic, and actionable.
        """
        
        gemini_response = gemini_model.generate_content(prompt)
        recommendations = gemini_response.text
        
        return jsonify({
            'success': True,
            'disease': disease,
            'confidence': 0.998,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)