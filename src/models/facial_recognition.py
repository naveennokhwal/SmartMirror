import torch
import cv2
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from PIL import Image

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
DATASET_PATH = "C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\SmartMirror\data"  # Change to your dataset folder

# Dictionary to store face embeddings
face_db = {}

# Preprocess function
transform = transforms.Compose([
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
            if box is not None:
                print(box)
                print('\n')
                x1, y1, x2, y2 = map(int, box.squeeze())
                cropped_face = img_np[y1:y2, x1:x2] if y1 < y2 and x1 < x2 else None
                cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB))  # Convert NumPy to PIL
                cropped_face_tensor = transform(cropped_face_pil)  # Now apply the transform
                cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    embedding = facenet(cropped_face_tensor).cpu().numpy()

                if embedding is None:
                    print(f"Embedding not generated for image")

                embeddings.append(embedding)
        
        # Store the mean embedding of the person
        face_db[person] = np.mean(embeddings, axis=0)
print("Face database built successfully!")

# Step 2: Real-Time Face Recognition
cap = cv2.VideoCapture(0)  # Start webcam

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame.")
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2] if y1 < y2 and x1 < x2 else None
            
            if face is None or face.size == 0:
                print("No face detected. Skipping frame...")
                continue
            
            face_resized = cv2.resize(face, (160, 160))
            face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))  # Convert NumPy to PIL
            face_tensor = transform(face_pil)  # Now apply the transform
            face_tensor = face_tensor.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                face_embedding = facenet(face_tensor).cpu().numpy()
            
            if face_embedding is None:
                print(f"Embedding not generated for image {img_path}")


            # Find best match using cosine similarity
            best_match = "Unknown"
            max_similarity = 0.0

            for name, db_embedding in face_db.items():
                similarity = cosine_similarity(torch.tensor(face_embedding), torch.tensor(db_embedding)).item()
                print(f"Similarity with {name}: {similarity}")
                if similarity > max_similarity and similarity > 0.65:  # Threshold for recognition
                    max_similarity = similarity
                    best_match = name

            # Draw bounding box and label
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display output
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()