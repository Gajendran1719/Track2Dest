# AI-Powered Smart Campus Navigation System

## Project Overview
A full-stack web application that uses AI/ML (simulated CNN) + Dijkstra's Algorithm 
to help users navigate a college campus by uploading a photo of their location.

## Technology Stack
- **Backend**: Python 3, Flask, SQLAlchemy
- **AI/ML**: Simulated CNN (MobileNetV2/ResNet50 architecture), Pillow, NumPy
- **Algorithm**: Dijkstra's Shortest Path Algorithm
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript, SVG Canvas

## Modules Implemented
1. User Registration & Login (with password hashing)
2. Image Upload (drag & drop, validation, preview)
3. AI Landmark Recognition (CNN simulation with confidence scores)
4. Location Detection (maps landmark to campus map)
5. Destination Selection (15 campus locations)
6. Intelligent Route Planning (Dijkstra's Algorithm)
7. Navigation Guide (turn-by-turn, distance, ETA)
8. Database (SQLite - users, history, feedback)
9. Feedback & Model Improvement (training loop simulation)

## Campus Locations
Main Gate, Admin Block, Library, CSE Block, ECE Block, Science Block, 
Auditorium, Canteen, Parking, Sports Ground, Principal Office, Exam Cell, 
Labs, Hostel, Bus Stop

## Setup & Run
```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:5050
```

## Project Structure
```
campus-nav/
├── app.py              # Main Flask application (all routes + AI logic)
├── requirements.txt    # Python dependencies
├── templates/
│   ├── base.html       # Base template with navigation
│   ├── index.html      # Landing page
│   ├── register.html   # Registration
│   ├── login.html      # Login
│   ├── dashboard.html  # User dashboard
│   ├── navigate.html   # Main navigation page (core feature)
│   ├── history.html    # Navigation history
│   ├── feedback.html   # AI feedback form
│   └── profile.html    # User profile & stats
└── static/
    └── images/uploads/ # Uploaded campus images
```

## Key Features
- No GPS required - uses image recognition
- Interactive SVG campus map with animated route
- Real-time confidence score display (bar chart)
- Turn-by-turn navigation instructions
- Navigation history tracking
- Feedback loop for AI improvement
