from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, json, math, random, base64
from datetime import datetime
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'campus_nav_secret_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///campus.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/images/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# ─── Models ───────────────────────────────────────────────────────────────────

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='student')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class NavigationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    from_location = db.Column(db.String(100))
    to_location = db.Column(db.String(100))
    distance = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    predicted_location = db.Column(db.String(100))
    correct_location = db.Column(db.String(100))
    rating = db.Column(db.Integer)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ─── Campus Graph Data ─────────────────────────────────────────────────────────

CAMPUS_NODES = {
    "Main Gate":        {"x": 500, "y": 780, "type": "gate",     "color": "#e74c3c"},
    "Admin Block":      {"x": 500, "y": 640, "type": "building", "color": "#3498db"},
    "Library":          {"x": 320, "y": 500, "type": "building", "color": "#9b59b6"},
    "CSE Block":        {"x": 680, "y": 500, "type": "building", "color": "#2ecc71"},
    "ECE Block":        {"x": 780, "y": 380, "type": "building", "color": "#f39c12"},
    "Science Block":    {"x": 220, "y": 380, "type": "building", "color": "#1abc9c"},
    "Auditorium":       {"x": 500, "y": 380, "type": "building", "color": "#e67e22"},
    "Canteen":          {"x": 350, "y": 640, "type": "amenity",  "color": "#e74c3c"},
    "Parking":          {"x": 650, "y": 700, "type": "amenity",  "color": "#7f8c8d"},
    "Sports Ground":    {"x": 150, "y": 550, "type": "facility", "color": "#27ae60"},
    "Principal Office": {"x": 500, "y": 560, "type": "office",   "color": "#8e44ad"},
    "Exam Cell":        {"x": 620, "y": 420, "type": "office",   "color": "#c0392b"},
    "Labs":             {"x": 750, "y": 500, "type": "facility", "color": "#2980b9"},
    "Hostel":           {"x": 180, "y": 300, "type": "facility", "color": "#16a085"},
    "Bus Stop":         {"x": 500, "y": 870, "type": "gate",     "color": "#95a5a6"},
}

CAMPUS_EDGES = [
    ("Main Gate", "Admin Block", 140),
    ("Main Gate", "Parking", 150),
    ("Main Gate", "Bus Stop", 90),
    ("Admin Block", "Library", 180),
    ("Admin Block", "CSE Block", 180),
    ("Admin Block", "Principal Office", 80),
    ("Admin Block", "Canteen", 130),
    ("Library", "Science Block", 160),
    ("Library", "Auditorium", 200),
    ("Library", "Sports Ground", 180),
    ("CSE Block", "ECE Block", 160),
    ("CSE Block", "Auditorium", 180),
    ("CSE Block", "Labs", 100),
    ("CSE Block", "Exam Cell", 130),
    ("Science Block", "Auditorium", 180),
    ("Science Block", "Hostel", 170),
    ("Auditorium", "Exam Cell", 140),
    ("ECE Block", "Labs", 130),
    ("Hostel", "Sports Ground", 200),
    ("Canteen", "Parking", 190),
    ("Principal Office", "Exam Cell", 140),
]

# Build adjacency list
def build_graph():
    graph = {node: [] for node in CAMPUS_NODES}
    for u, v, w in CAMPUS_EDGES:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph

GRAPH = build_graph()

# ─── Simulated AI Landmark Recognition ───────────────────────────────────────

LANDMARK_KEYWORDS = {
    "Main Gate":        ["gate", "entrance", "main", "security"],
    "Library":          ["library", "book", "reading", "shelves"],
    "CSE Block":        ["computer", "cse", "lab", "server"],
    "ECE Block":        ["electronics", "ece", "circuit"],
    "Science Block":    ["science", "physics", "chemistry", "biology"],
    "Auditorium":       ["auditorium", "hall", "stage", "event"],
    "Canteen":          ["canteen", "food", "cafeteria", "eat"],
    "Admin Block":      ["admin", "office", "reception", "administration"],
    "Parking":          ["parking", "vehicle", "car", "bike"],
    "Sports Ground":    ["sports", "ground", "field", "play"],
    "Principal Office": ["principal", "director", "head"],
    "Exam Cell":        ["exam", "examination", "test", "hall"],
    "Labs":             ["laboratory", "lab", "experiment"],
    "Hostel":           ["hostel", "dormitory", "room", "stay"],
    "Bus Stop":         ["bus", "stop", "transport"],
}

def analyze_image_colors(image_path):
    """Extract dominant color features from image for simulated AI."""
    try:
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        arr = np.array(img, dtype=float)
        r_mean = arr[:,:,0].mean()
        g_mean = arr[:,:,1].mean()
        b_mean = arr[:,:,2].mean()
        brightness = (r_mean + g_mean + b_mean) / 3
        return {"r": r_mean, "g": g_mean, "b": b_mean, "brightness": brightness}
    except:
        return {"r": 128, "g": 128, "b": 128, "brightness": 128}

def simulate_cnn_recognition(image_path, filename):
    """Simulate CNN landmark recognition using filename hints + image analysis."""
    colors = analyze_image_colors(image_path)
    fname_lower = filename.lower().replace('_', ' ').replace('-', ' ')
    
    scores = {}
    for landmark, keywords in LANDMARK_KEYWORDS.items():
        score = random.uniform(0.05, 0.2)
        for kw in keywords:
            if kw in fname_lower:
                score += 0.4
        scores[landmark] = round(min(score, 0.99), 3)
    
    # Normalize to make top score between 0.75–0.95
    top_key = max(scores, key=scores.get)
    if scores[top_key] < 0.3:
        winner = random.choice(list(CAMPUS_NODES.keys()))
        scores[winner] = round(random.uniform(0.78, 0.93), 3)
        top_key = winner

    total = sum(scores.values())
    scores = {k: round(v / total, 3) for k, v in scores.items()}
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return top_key, sorted_scores

# ─── Path Planning (Dijkstra) ─────────────────────────────────────────────────

def dijkstra(graph, start, end):
    import heapq
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == end: break
        for v, w in graph[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    
    path, node = [], end
    while node:
        path.append(node)
        node = prev[node]
    path.reverse()
    
    if path[0] != start:
        return [], float('inf')
    return path, dist[end]

def generate_instructions(path):
    """Generate turn-by-turn instructions for a path."""
    if len(path) < 2:
        return ["You are already at your destination."]
    
    instructions = [f"📍 Start at <strong>{path[0]}</strong>"]
    directions = ["Head towards", "Walk to", "Continue to", "Proceed to", "Turn towards"]
    
    for i in range(1, len(path) - 1):
        action = random.choice(directions)
        instructions.append(f"➡️ {action} <strong>{path[i]}</strong>")
    
    instructions.append(f"🏁 You have arrived at <strong>{path[-1]}</strong>")
    return instructions

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'student')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return render_template('register.html')
        
        user = User(
            name=name, email=email,
            password=generate_password_hash(password), role=role
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_role'] = user.role
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    history = NavigationHistory.query.filter_by(user_id=user.id).order_by(NavigationHistory.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', user=user, history=history)

@app.route('/navigate')
def navigate():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    import json
    destinations = list(CAMPUS_NODES.keys())
    return render_template('navigate.html', 
        destinations=destinations,
        nodes_json=json.dumps(CAMPUS_NODES),
        edges_json=json.dumps([[u,v,w] for u,v,w in CAMPUS_EDGES]))

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    location, scores = simulate_cnn_recognition(save_path, filename)
    top5 = dict(list(scores.items())[:5])
    confidence = scores[location]
    
    return jsonify({
        'detected_location': location,
        'confidence': confidence,
        'top_predictions': top5,
        'node_data': CAMPUS_NODES[location]
    })

@app.route('/api/navigate', methods=['POST'])
def api_navigate():
    data = request.get_json()
    src = data.get('from')
    dst = data.get('to')
    
    if src not in CAMPUS_NODES or dst not in CAMPUS_NODES:
        return jsonify({'error': 'Invalid location'}), 400
    if src == dst:
        return jsonify({'error': 'Source and destination are the same'}), 400
    
    path, distance = dijkstra(GRAPH, src, dst)
    if not path:
        return jsonify({'error': 'No route found'}), 404
    
    instructions = generate_instructions(path)
    speed_mpm = 80  # metres per minute walking
    eta = round(distance / speed_mpm, 1)
    
    # Save history
    if 'user_id' in session:
        nav = NavigationHistory(
            user_id=session['user_id'],
            from_location=src, to_location=dst, distance=distance
        )
        db.session.add(nav)
        db.session.commit()
    
    return jsonify({
        'path': path,
        'distance': round(distance),
        'eta_minutes': eta,
        'instructions': instructions,
        'nodes': {n: CAMPUS_NODES[n] for n in path},
        'edges': CAMPUS_EDGES
    })

@app.route('/api/campus_map')
def api_campus_map():
    return jsonify({'nodes': CAMPUS_NODES, 'edges': CAMPUS_EDGES})

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        fb = Feedback(
            user_id=session['user_id'],
            predicted_location=request.form['predicted'],
            correct_location=request.form['correct'],
            rating=int(request.form['rating']),
            comment=request.form.get('comment', '')
        )
        db.session.add(fb)
        db.session.commit()
        flash('Thank you for your feedback! This helps improve our AI model.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('feedback.html', locations=list(CAMPUS_NODES.keys()))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    records = NavigationHistory.query.filter_by(user_id=session['user_id']).order_by(NavigationHistory.timestamp.desc()).all()
    return render_template('history.html', records=records)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    total_nav = NavigationHistory.query.filter_by(user_id=user.id).count()
    total_dist = db.session.query(db.func.sum(NavigationHistory.distance)).filter_by(user_id=user.id).scalar() or 0
    return render_template('profile.html', user=user, total_nav=total_nav, total_dist=round(total_dist))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5050)
