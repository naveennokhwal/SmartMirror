import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from base64 import b64decode
import io
from PIL import Image

# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)  # 7 emotion classes
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 6 * 6)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
               4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Initialize the model and face cascade detector
class EmotionDetector:
    def __init__(self, model_path=r'api\final_emotion_model.pth'):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model
            self.model = EmotionCNN()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            # Load face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"Error initializing EmotionDetector: {str(e)}")
            traceback.print_exc()
            raise

    
    def detect_from_image(self, image_data):
        """
        Process an image and return the detected emotion
        
        Args:
            image_data: Can be a numpy array, PIL Image, or base64 encoded string
            
        Returns:
            emotion_text: String representing the detected emotion
            face_detected: Boolean indicating if a face was detected
        """
        # Convert input to numpy array if it's not already
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle base64 encoded image
            image_data = image_data.split(',')[1]
            image_bytes = b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)
        elif isinstance(image_data, str):
            # Assume it's base64 without data URI
            image_bytes = b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)
        elif isinstance(image_data, Image.Image):
            # PIL Image
            frame = np.array(image_data)
        else:
            # Assume it's already a numpy array
            frame = image_data
            
        # Convert to grayscale if the image is in color
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray)
        
        if len(faces) == 0:
            return "No Face", False
            
        # Process the first face found
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to match the model's expected input
        roi_gray_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_tensor = torch.FloatTensor(roi_gray_resized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(roi_tensor)
            emotion_label = torch.argmax(output).item()
            emotion_text = emotion_map[emotion_label]
            
        return emotion_text