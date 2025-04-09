import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU for debugging

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from utils.image_processing import preprocess_for_leaf_detection
from utils.disease_info import DISEASE_INFO
from datetime import datetime
import pandas as pd
from sqlalchemy import func

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
INSTANCE_FOLDER = r"/home/ubuntuwsl/Tomato_Leaf_Disease_WebApp/instance"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{INSTANCE_FOLDER}/site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

MODEL_PATH = r"/home/ubuntuwsl/Tomato_Leaf_Disease_WebApp/models/mobilenetv2_finetuned_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
MIN_CONTOUR_AREA = 500
MIN_CONFIDENCE_THRESHOLD = 85.0

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

# Initialize Flask-SQLAlchemy
db = SQLAlchemy(app)

# Load MobileNetV2 model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise ValueError(f"Failed to load model: {str(e)}")

class_names = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus", "healthy", "powdery_mildew"
]

# Location data
COUNTRIES = ['India', 'Other']
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
]

state_districts = {
    "Kerala": [
        "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode",
        "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"
    ],
    "Karnataka": [
        "Bagalkot", "Bangalore Rural", "Bangalore Urban", "Belgaum", "Bellary", "Bidar", "Chamarajanagar",
        "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davanagere", "Dharwad",
        "Gadag", "Gulbarga", "Hassan", "Haveri", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysore",
        "Raichur", "Ramanagara", "Shimoga", "Tumkur", "Udupi", "Uttara Kannada", "Yadgir"
    ],
    "Tamil Nadu": [
        "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kanchipuram",
        "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Nagapattinam", "Namakkal", "Nilgiris",
        "Perambalur", "Pudukkottai", "Ramanathapuram", "Salem", "Sivaganga", "Thanjavur", "Theni",
        "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tiruppur", "Tiruvallur", "Tiruvannamalai",
        "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"
    ],
    # Add more states and districts as needed
}

# Database Models
class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    country = db.Column(db.String(100), nullable=True)
    state = db.Column(db.String(100), nullable=True)
    district = db.Column(db.String(100), nullable=True)
    flagged = db.Column(db.Boolean, default=False)

# Create the database
with app.app_context():
    try:
        db.create_all()
        if not Admin.query.first():
            admin = Admin(username='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

def test_classification():
    dummy_img = np.random.rand(224, 224, 3) * 255
    dummy_img = dummy_img.astype(np.uint8)
    cv2.imwrite("static/uploads/dummy.jpg", dummy_img)
    try:
        prediction = classify_leaf("uploads/dummy.jpg")
        print(f"Dummy image prediction: {prediction}")
    except Exception as e:
        print(f"Error classifying dummy image: {str(e)}")

test_classification()

def detect_leaves(image_path: str) -> tuple:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    edges = preprocess_for_leaf_detection(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.9:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h != 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    padding = 10
                    x1 = max(x - padding, 0)
                    y1 = max(y - padding, 0)
                    x2 = min(x + w + padding, img.shape[1])
                    y2 = min(x + h + padding, img.shape[0])
                    leaf_boxes.append((x1, y1, x2, y2))

    if not leaf_boxes:
        return [], [], img

    leaf_boxes = sorted(leaf_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    cropped_paths = []
    for i, (x1, y1, x2, y2) in enumerate(leaf_boxes):
        leaf_region = img[y1:y2, x1:x2]
        if leaf_region.size == 0:
            continue
        leaf_region_resized = cv2.resize(leaf_region, IMG_SIZE, interpolation=cv2.INTER_AREA)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{image_name}leaf{i}.jpg")
        if cv2.imwrite(output_path, leaf_region_resized):
            cropped_paths.append(f"uploads/{image_name}leaf{i}.jpg")

    return leaf_boxes, cropped_paths, img

def classify_leaf(cropped_path: str) -> dict:
    try:
        img_path = os.path.join('static', cropped_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        if img_array.shape[-1] != 3:
            raise ValueError(f"Image must have 3 channels (RGB), but has {img_array.shape[-1]} channels.")
        if img_array.shape[0] != IMG_SIZE[0] or img_array.shape[1] != IMG_SIZE[1]:
            raise ValueError(f"Image dimensions must be {IMG_SIZE}, but got {img_array.shape[:2]}.")

        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0] * 100

        return {
            "label": class_names[predicted_class],
            "confidence": round(confidence, 2),
            "disease_info": DISEASE_INFO.get(class_names[predicted_class], {
                "description": "No description available.",
                "treatment": "No treatment information available."
            })
        }
    except Exception as e:
        raise ValueError(f"Error classifying image: {str(e)}")

@app.route('/')
def index():
    selected_country = request.args.get('country', 'India')
    selected_state = request.args.get('state', '')
    selected_district = request.args.get('district', '')
    districts = state_districts.get(selected_state, []) if selected_country == 'India' and selected_state else []
    return render_template('index.html', page='upload', countries=COUNTRIES, indian_states=INDIAN_STATES,
                           districts=districts, state_districts=state_districts,
                           selected_country=selected_country, selected_state=selected_state,
                           selected_district=selected_district)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded.", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=[], state_districts=state_districts,
                               selected_country='India', selected_state='', selected_district='')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No image selected.", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=[], state_districts=state_districts,
                               selected_country='India', selected_state='', selected_district='')

    country = request.form.get('country', 'India').strip()
    state = request.form.get('state', '').strip()
    district = request.form.get('district', '').strip()

    if not country:
        return render_template('index.html', error="Please select a country.", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=[], state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)
    if country == 'India' and not state:
        return render_template('index.html', error="Please select a state for India.", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=[], state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)
    if country == 'India' and state and state in state_districts and not district:
        return render_template('index.html', error="Please select a district for the selected state.", page='upload',
                               countries=COUNTRIES, indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)
    if country == 'India' and state and district and district not in state_districts.get(state, []):
        return render_template('index.html', error="Invalid district for the selected state.", page='upload',
                               countries=COUNTRIES, indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(image_path)
        if not os.path.exists(image_path):
            raise OSError("Failed to save uploaded file.")
    except Exception as e:
        return render_template('index.html', error=f"Error saving image: {str(e)}", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)

    try:
        leaf_boxes, cropped_paths, original_img = detect_leaves(image_path)
        if not leaf_boxes:
            return render_template('index.html', error="No leaves detected.", page='upload', countries=COUNTRIES,
                                   indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                                   state_districts=state_districts,
                                   selected_country=country, selected_state=state, selected_district=district)

        if len(cropped_paths) == 1:
            prediction = classify_leaf(cropped_paths[0])
            prediction['image_path'] = cropped_paths[0]
            if prediction['confidence'] < MIN_CONFIDENCE_THRESHOLD:
                return render_template('index.html', page='upload', error="Prediction confidence below 85%.",
                                       countries=COUNTRIES, indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                                       state_districts=state_districts,
                                       selected_country=country, selected_state=state, selected_district=district)
            activity = UserActivity(
                image_path=prediction['image_path'],
                predicted_disease=prediction['label'],
                confidence=prediction['confidence'],
                country=country,
                state=state if country == 'India' else None,
                district=district if country == 'India' and state in state_districts else None
            )
            db.session.add(activity)
            db.session.commit()
            return render_template('index.html', page='result', prediction=prediction, countries=COUNTRIES,
                                   indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                                   state_districts=state_districts,
                                   selected_country=country, selected_state=state, selected_district=district)

        return render_template('select_leaf.html', original_image=f"uploads/{filename}", leaves=zip(cropped_paths, leaf_boxes),
                               country=country, state=state, district=district)

    except Exception as e:
        db.session.rollback()
        return render_template('index.html', error=f"Error processing image: {str(e)}", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)

@app.route('/classify_leaf', methods=['POST'])
def classify_selected_leaf():
    selected_leaf_path = request.form.get('leaf_path')
    country = request.form.get('country', 'India').strip()
    state = request.form.get('state', '').strip()
    district = request.form.get('district', '').strip()
    if not selected_leaf_path:
        return render_template('index.html', error="No leaf selected.", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)

    try:
        prediction = classify_leaf(selected_leaf_path)
        prediction['image_path'] = selected_leaf_path
        if prediction['confidence'] < MIN_CONFIDENCE_THRESHOLD:
            return render_template('index.html', page='upload', error="Prediction confidence below 85%.",
                                   countries=COUNTRIES, indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                                   state_districts=state_districts,
                                   selected_country=country, selected_state=state, selected_district=district)
        activity = UserActivity(
            image_path=prediction['image_path'],
            predicted_disease=prediction['label'],
            confidence=prediction['confidence'],
            country=country,
            state=state if country == 'India' else None,
            district=district if country == 'India' and state in state_districts else None
        )
        db.session.add(activity)
        db.session.commit()
        return render_template('index.html', page='result', prediction=prediction, countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)
    except Exception as e:
        db.session.rollback()
        return render_template('index.html', error=f"Error classifying leaf: {str(e)}", page='upload', countries=COUNTRIES,
                               indian_states=INDIAN_STATES, districts=state_districts.get(state, []),
                               state_districts=state_districts,
                               selected_country=country, selected_state=state, selected_district=district)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        admin = Admin.query.filter_by(username=username).first()

        if admin and admin.check_password(password):
            login_user(admin)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('admin_login.html')

@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    date_start = request.args.get('date_start')
    date_end = request.args.get('date_end')
    disease = request.args.get('disease')
    confidence_min = request.args.get('confidence_min', type=float)
    confidence_max = request.args.get('confidence_max', type=float)
    country = request.args.get('country', 'India')
    state = request.args.get('state')
    district = request.args.get('district')
    graph_type = request.args.get('graph_type', 'bar')

    query = UserActivity.query

    if country:
        query = query.filter(UserActivity.country == country)
    if state and country == 'India':
        query = query.filter(UserActivity.state == state)
    if district and country == 'India' and state in state_districts:
        query = query.filter(UserActivity.district == district)
    if date_start:
        query = query.filter(UserActivity.timestamp >= date_start)
    if date_end:
        query = query.filter(UserActivity.timestamp <= date_end)
    if disease:
        query = query.filter(UserActivity.predicted_disease == disease)
    if confidence_min is not None:
        query = query.filter(UserActivity.confidence >= confidence_min)
    if confidence_max is not None:
        query = query.filter(UserActivity.confidence <= confidence_max)

    activities = query.order_by(UserActivity.timestamp.desc()).all()

    disease_counts = query.with_entities(UserActivity.predicted_disease, func.count(UserActivity.id)).group_by(UserActivity.predicted_disease).all()
    disease_data = {disease: count for disease, count in disease_counts}

    confidence_stats = {
        'avg': query.with_entities(func.avg(UserActivity.confidence)).scalar() or 0,
        'min': query.with_entities(func.min(UserActivity.confidence)).scalar() or 0,
        'max': query.with_entities(func.max(UserActivity.confidence)).scalar() or 0,
        'low_count': query.filter(UserActivity.confidence < MIN_CONFIDENCE_THRESHOLD).count() or 0
    }

    if country == 'India':
        if district:
            location_trends_raw = query.with_entities(
                UserActivity.district,
                UserActivity.predicted_disease,
                func.count(UserActivity.id).label('count')
            ).filter(UserActivity.district == district).group_by(UserActivity.district, UserActivity.predicted_disease).all()
        elif state:
            location_trends_raw = query.with_entities(
                UserActivity.district,
                UserActivity.predicted_disease,
                func.count(UserActivity.id).label('count')
            ).filter(UserActivity.state == state).group_by(UserActivity.district, UserActivity.predicted_disease).all()
        else:
            location_trends_raw = query.with_entities(
                UserActivity.state,
                UserActivity.predicted_disease,
                func.count(UserActivity.id).label('count')
            ).filter(UserActivity.country == 'India').group_by(UserActivity.state, UserActivity.predicted_disease).all()
    else:
        location_trends_raw = query.with_entities(
            UserActivity.country,
            UserActivity.predicted_disease,
            func.count(UserActivity.id).label('count')
        ).filter(UserActivity.country == country).group_by(UserActivity.country, UserActivity.predicted_disease).all()

    location_trends_dict = {}
    for location, disease, count in location_trends_raw:
        if location not in location_trends_dict:
            location_trends_dict[location] = {}
        location_trends_dict[location][disease] = count

    location_trends = []
    for location, diseases in location_trends_dict.items():
        total_count = sum(diseases.values())
        sorted_diseases = sorted(diseases.items(), key=lambda x: x[1], reverse=True)
        most_reported = sorted_diseases[0][0] if sorted_diseases else None
        location_trends.append({
            'location': f"{location} ({total_count})",
            'diseases': [{'name': d, 'count': c, 'is_most_reported': d == most_reported} for d, c in sorted_diseases]
        })

    num_users = query.count() or 0
    districts = state_districts.get(state, []) if country == 'India' and state else []

    return render_template('admin_dashboard.html',
                           activities=activities,
                           disease_data=disease_data,
                           confidence_stats=confidence_stats,
                           location_trends=location_trends,
                           num_users=num_users,
                           class_names=class_names,
                           graph_type=graph_type,
                           countries=COUNTRIES,
                           indian_states=INDIAN_STATES,
                           districts=districts,
                           state_districts=state_districts,
                           selected_country=country,
                           selected_state=state or '',
                           selected_district=district or '')

@app.route('/admin/flag_activity/<int:id>', methods=['POST'])
@login_required
def flag_activity(id):
    activity = UserActivity.query.get_or_404(id)
    activity.flagged = not activity.flagged
    db.session.commit()
    flash(f"Activity {id} {'flagged' if activity.flagged else 'unflagged'} for review.", 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_activity/<int:id>', methods=['POST'])
@login_required
def delete_activity(id):
    activity = UserActivity.query.get_or_404(id)
    db.session.delete(activity)
    db.session.commit()
    flash(f"Activity record with ID {id} has been deleted.", 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/download_activities', methods=['GET'])
@login_required
def download_activities():
    activities = UserActivity.query.all()
    data = [{
        'ID': a.id,
        'Image Path': a.image_path,
        'Predicted Disease': a.predicted_disease,
        'Confidence': a.confidence,
        'Timestamp': a.timestamp,
        'Country': a.country,
        'State': a.state,
        'District': a.district,
        'Flagged': a.flagged
    } for a in activities]
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=user_activities.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)