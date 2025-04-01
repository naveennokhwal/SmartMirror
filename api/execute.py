import torch
import cv2
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model for emotion recognition
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Load pre-trained FaceNet model for face recognition
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load emotion recognition model
emotion_model = EmotionCNN().to(device)
emotion_model.load_state_dict(torch.load(r'api\final_emotion_model.pth', map_location=device))
emotion_model.eval()

# Path to dataset for face recognition
DATASET_PATH = r"data"  # Change to your dataset folder

# Dictionary to store face embeddings
face_db = {}

# Emotion mapping
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
               4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Preprocess function for face recognition
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(image_path):
    """ Load and preprocess an image for FaceNet embedding """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_pil = Image.fromarray(img_rgb)
    return img_rgb, img_pil

# Step 1: Create Face Database
print("Building face database...")
for person in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH, person)
    if os.path.isdir(person_folder):
        embeddings = []
        for img_file in os.listdir(person_folder):
            if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(person_folder, img_file)
            img_np, img_PIL = preprocess_image(img_path)

            box, _ = mtcnn.detect(img_PIL)
            if box is not None and len(box) > 0:
                x1, y1, x2, y2 = map(int, box[0])
                if y1 < y2 and x1 < x2:
                    cropped_face = img_np[y1:y2, x1:x2]
                    cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                    cropped_face_pil = Image.fromarray(cropped_face_resized)
                    cropped_face_tensor = face_transform(cropped_face_pil)
                    cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = facenet(cropped_face_tensor).cpu().numpy()
                    
                    embeddings.append(embedding)
        
        if embeddings:
            # Store the mean embedding of the person
            face_db[person] = np.mean(embeddings, axis=0)
            print(f"Added {person} to face database")
        else:
            print(f"No valid embeddings found for {person}")

print("Face database built successfully!")

# Step 2: Real-Time Face Recognition and Emotion Detection
cap = cv2.VideoCapture(0)  # Start webcam

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame.")
        break
    
    # Convert to RGB for MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, _ = mtcnn.detect(img_rgb)
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid box coordinates
            if y1 >= y2 or x1 >= x2 or y1 < 0 or x1 < 0 or y2 > frame.shape[0] or x2 > frame.shape[1]:
                continue
                
            face = frame[y1:y2, x1:x2]
            
            # Face Recognition
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = face_transform(face_pil)
            face_tensor = face_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                face_embedding = facenet(face_tensor).cpu().numpy()
            
            # Find best match using cosine similarity
            best_match = "Unknown"
            max_similarity = 0.0

            for name, db_embedding in face_db.items():
                similarity = cosine_similarity(torch.tensor(face_embedding), torch.tensor(db_embedding)).item()
                if similarity > max_similarity and similarity > 0.65:  # Threshold for recognition
                    max_similarity = similarity
                    best_match = name
            
            # Emotion Recognition
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray_resized = cv2.resize(face_gray, (48, 48)) / 255.0
            emotion_tensor = torch.FloatTensor(face_gray_resized).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                emotion_output = emotion_model(emotion_tensor)
                emotion_label = torch.argmax(emotion_output).item()
                emotion_text = emotion_map[emotion_label]
            
            # Draw bounding box
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display name and emotion
            name_text = f"Name: {best_match}"
            emotion_display = f"Emotion: {emotion_text}"
            
            cv2.putText(frame, name_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, emotion_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display output
    cv2.imshow("Face Recognition and Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
