from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys

# Import functions from API modules
try:
    from api.identify import FaceRecognizer
    face_recognizer = FaceRecognizer()
    from api.emotion import EmotionDetector
    emotion_detector = EmotionDetector()

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Default route to check if the server is running
@app.route('/')
def home():
    return "Flask server is running!"

# Route for user recognition
@app.route('/identify', methods=['POST'])
def identify():
    try:
        print("Received identify request")
        frame = request.json.get('image')  # Changed 'frame' to 'image' to match frontend
        if not frame:
            return jsonify({"error": "No image data provided"}), 400
        
        print("Calling recognize_user function")
        user = face_recognizer.recognize_user(frame)  # Call the function from identify.py
        print(f"Recognition result: {user}")
        return jsonify({"User": user})
    except Exception as e:
        print(f"Error in /identify endpoint: {str(e)}")
        traceback.print_exc()  # Print the full traceback
        return jsonify({"error": str(e)}), 500

# Route for mood detection
@app.route('/emotion', methods=['POST'])
def emotion():
    try:
        print("Received emotion request")
        frame = request.json.get('image')  # Changed 'frame' to 'image' to match frontend
        if not frame:
            return jsonify({"error": "No image data provided"}), 400
        
        print("Calling detect_from_image method")
        mood = emotion_detector.detect_from_image(frame)  # Call the function from emotion.py
        print(f"Emotion result: {mood}")
        return jsonify({"Mood": mood})
    except Exception as e:
        print(f"Error in /emotion endpoint: {str(e)}")
        traceback.print_exc()  # Print the full traceback
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
