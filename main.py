from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Create the database and the tables
with app.app_context():
    db.create_all()

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get form data
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])
            
            # Create input array
            user_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            user_data_scaled = scaler.transform(user_data)
            
            # Make prediction
            prediction = model.predict(user_data_scaled)
            
            # Interpret the result
            if prediction[0] == 1:
                result = "Based on the provided data, you have a high chance of having diabetes."
            else:
                result = "Based on the provided data, you have a low chance of having diabetes."
        except Exception as e:
            result = f"Error in processing input: {str(e)}"
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result='')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            return redirect(url_for('login'))
        

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
