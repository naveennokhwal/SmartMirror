import torch
import cv2
import numpy as np
import os
import base64
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
import io

class FaceRecognizer:
    def __init__(self, dataset_path= r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\SmartMirror_new\data"):
        """
        Initialize the face recognition system
        
        Args:
            dataset_path (str): Path to the directory containing face images organized in folders by person
        """
        print("Initializing FaceRecognizer...")
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        print("MTCNN loaded successfully")
        
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print("FaceNet loaded successfully")
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Dataset path
        self.dataset_path = dataset_path
        
        # Build face database
        self.face_db = {}
        self.build_face_database()
        
        # Recognition threshold
        self.similarity_threshold = 0.65
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image from file for FaceNet embedding
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            tuple: (numpy array, PIL Image) of the processed image
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_pil = Image.fromarray(img_rgb)
        return img_rgb, img_pil
    
    def decode_base64_image(self, base64_image):
        """
        Decode a base64 encoded image to a numpy array
        
        Args:
            base64_image (str): Base64 encoded image string
        
        Returns:
            numpy.ndarray: Decoded image as a numpy array
        """
        frame_data = base64.b64decode(base64_image.split(',')[1])
        np_img = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        return frame
        
    def build_face_database(self):
        """
        Build a database of face embeddings from the dataset
        """
        print("Building face database...")
        for person in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person)
            if os.path.isdir(person_folder):
                embeddings = []
                for img_file in os.listdir(person_folder):
                    if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    img_path = os.path.join(person_folder, img_file)
                    
                    try:
                        img_np, img_PIL = self.preprocess_image(img_path)
                        embedding = self.extract_embedding_from_image(img_PIL)
                        if embedding is not None:
                            embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                
                if embeddings:
                    self.face_db[person] = np.mean(embeddings, axis=0)
        
        print(f"Face database built successfully with {len(self.face_db)} individuals!")
    
    def extract_embedding_from_image(self, img_pil):
        """
        Extract face embedding from a PIL image
        
        Args:
            img_pil (PIL.Image): Image to process
        
        Returns:
            numpy.ndarray: Face embedding or None if face detection failed
        """
        boxes, _ = self.mtcnn.detect(img_pil)
        
        if boxes is not None and len(boxes) > 0:
            try:
                x1, y1, x2, y2 = map(int, boxes[0])
                if y1 < y2 and x1 < x2:
                    # Convert PIL image to numpy array for cropping
                    img_np = np.array(img_pil)
                    cropped_face = img_np[y1:y2, x1:x2]
                    cropped_face_resized = cv2.resize(cropped_face, (160, 160))
                    cropped_face_pil = Image.fromarray(cropped_face_resized)
                    cropped_face_tensor = self.transform(cropped_face_pil)
                    cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.facenet(cropped_face_tensor).cpu().numpy()
                        return embedding
            except Exception as e:
                print(f"Error extracting embedding: {str(e)}")
        
        return None
    
    def recognize_face(self, face_embedding):
        """
        Recognize a face by comparing its embedding to the database
        
        Args:
            face_embedding (numpy.ndarray): Face embedding to recognize
        
        Returns:
            str: Name of the recognized person or "Unknown"
        """
        best_match = "Unknown"
        max_similarity = 0.0
        
        for name, db_embedding in self.face_db.items():
            similarity = cosine_similarity(
                torch.tensor(face_embedding), 
                torch.tensor(db_embedding)
            ).item()
            
            print(f"Similarity with {name}: {similarity}")
            
            if similarity > max_similarity and similarity > self.similarity_threshold:
                max_similarity = similarity
                best_match = name
        
        return best_match
    
    def recognize_user(self, frame_base64):
        """
        Recognize user from a base64 encoded frame
        
        Args:
            frame_base64 (str): Base64 encoded image
        
        Returns:
            str: Name of the recognized person or "Unknown"
        """
        try:
            print("Starting face recognition process")
            
            # Decode the base64-encoded image from the frontend
            frame = self.decode_base64_image(frame_base64)
            
            # Process the current frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(Image.fromarray(img_rgb))
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box)
                        if y1 < y2 and x1 < x2:
                            face = img_rgb[y1:y2, x1:x2]
                            face_resized = cv2.resize(face, (160, 160))
                            face_pil = Image.fromarray(face_resized)
                            face_tensor = self.transform(face_pil)
                            face_tensor = face_tensor.unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                face_embedding = self.facenet(face_tensor).cpu().numpy()
                            
                            return self.recognize_face(face_embedding)
                    except Exception as e:
                        print(f"Error processing detected face: {str(e)}")
            
            return "Unknown"
        except Exception as e:
            print(f"Error in recognize_user: {str(e)}")
            return f"Error: {str(e)}"

    def update_face_database(self):
        """
        Update the face database with new images
        """
        self.face_db = {}
        self.build_face_database()
        print("Face database updated successfully!")

