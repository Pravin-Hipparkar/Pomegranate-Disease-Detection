import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import os
import io

# --- 1. Model Definition (PomegranateCNN) ---
# This class must be identical to the model used for training.
class PomegranateCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PomegranateCNN, self).__init__()
        
        # Convolutional Block 1 (3 -> 16 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # Convolutional Block 2 (16 -> 32 channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # Size after two pooling layers: 224/4 = 56. Flatten size: 32 * 56 * 56
        self.flatten_size = 32 * 56 * 56 
        
        # Classifier (Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

# --- 2. Configuration and Preprocessing ---

# Define class names exactly as used during training
CLASS_NAMES = ['Anthracnose', 'Bacterial_Blight', 'Healthy']

# !!! FINAL DIRECT DOWNLOAD URL FOR YOUR GOOGLE DRIVE MODEL FILE !!!
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1ZHnrZUBFlGfITt8Es9Hq1MRVvJrGke5F" 

MODEL_PATH = 'pomegranate_cnn_best.pth' 
IMAGE_SIZE = 224

# Preprocessing: MUST match validation transforms used during training
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- 3. Model Loading with Download ---

@st.cache_resource
def load_model(model_path, download_url):
    """
    Loads the model weights, downloading them from Google Drive first if needed.
    """
    device = torch.device("cpu") # Use CPU for stable cloud deployment
    
    # 1. Download the file if it doesn't exist locally
    if not os.path.exists(model_path):
        st.info("Model file not found. Downloading weights from Google Drive...")
        
        try:
            # Use requests to download the content
            response = requests.get(download_url, stream=True)
            response.raise_for_status() # Check for bad status code
            
            # Write the content to the expected file path
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model weights downloaded successfully!")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model from external link: {e}")
            return None # Stop loading if download fails

    # 2. Load the downloaded weights into the model structure
    model = PomegranateCNN(num_classes=len(CLASS_NAMES))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # CRUCIAL: Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# --- 4. Prediction Function ---

def predict(image, model):
    """Predicts the class of a single PIL image."""
    try:
        # Apply transforms, add batch dimension (unsqueeze(0))
        input_tensor = TRANSFORMS(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            confidence, predicted_class_idx = torch.max(probabilities, 1)

        predicted_label = CLASS_NAMES[predicted_class_idx.item()]
        confidence_score = confidence.item() * 100
        
        all_probs = {name: prob.item() * 100 for name, prob in zip(CLASS_NAMES, probabilities[0])}

        return predicted_label, confidence_score, all_probs
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Prediction Error", 0.0, {}

# --- 5. Streamlit App Interface ---

def main():
    st.set_page_config(page_title="Pomegranate Disease Detector", layout="wide")
    
    st.title("🌿 Deep Learning-Based Pomegranate Disease Detection")
    st.markdown("Upload an image of a pomegranate fruit to predict its health status (Anthracnose, Bacterial Blight, or Healthy).")
    st.markdown("---")

    # Load the model once (triggers download if file is missing)
    model = load_model(MODEL_PATH, DOWNLOAD_URL)
    if model is None:
        return # Stop if model loading fails

    # File Uploader
    uploaded_file = st.file_uploader("Choose a Pomegranate Image...", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.markdown("### Prediction:")
            
            # Predict
            with st.spinner('Analyzing image...'):
                predicted_label, confidence_score, all_probs = predict(image, model)

            st.success(f"Result: **{predicted_label}**")
            st.metric("Confidence Score", f"{confidence_score:.2f}%")
            
        else:
            st.info("Awaiting image upload...")

    with col2:
        if uploaded_file is not None:
            st.subheader("Details and Scores")
            
            # Display all probabilities in a table
            if all_probs:
                prob_data = {
                    'Disease Class': all_probs.keys(),
                    'Probability (%)': [f"{v:.2f}" for v in all_probs.values()]
                }
                st.dataframe(prob_data, use_container_width=True, hide_index=True)
                
                # Recommendation 
                st.markdown("---")
                st.subheader("Recommendation:")
                if predicted_label == 'Healthy':
                    st.balloons()
                    st.markdown("✅ **The fruit appears healthy.** Continue routine monitoring.")
                else:
                    st.warning(f"🚨 **Disease Detected: {predicted_label}**")
                    st.markdown("It is recommended to apply appropriate **pesticide treatment** immediately to prevent spread and minimize crop loss.")

# Run the app
if __name__ == "__main__":
    # Ensure requests library is available before running main function
    try:
        import requests
        main()
    except ImportError:
        st.error("Missing libraries. Please install Streamlit, torch, torchvision, Pillow, and requests.")