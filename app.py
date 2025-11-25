import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import os
import io

# --- 1. Model Definition (PomegranateCNN) ---
class PomegranateCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PomegranateCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.flatten_size = 32 * 56 * 56 
        
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

# --- 2. Configuration ---
CLASS_NAMES = ['Anthracnose', 'Bacterial_Blight', 'Healthy']
MODEL_PATH = 'pomegranate_cnn_best.pth'
IMAGE_SIZE = 224
FILE_ID = "1ZHnrZUBFlGfITt8Es9Hq1MRVvJrGke5F"

TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- 3. Google Drive Downloader ---
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)

# --- 4. Model Loading ---
@st.cache_resource
def load_model(model_path, file_id):
    device = torch.device("cpu")
    
    if not os.path.exists(model_path):
        st.info(f"Downloading model weights... (ID: {file_id})")
        try:
            download_file_from_google_drive(file_id, model_path)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    model = PomegranateCNN(num_classes=len(CLASS_NAMES))
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.warning("File corrupted. Deleting and retrying...")
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

# --- 5. Prediction Function ---
def predict(image, model):
    try:
        input_tensor = TRANSFORMS(image).unsqueeze(0)
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
        return "Error", 0.0, {}

# --- 6. Main Interface ---
def main():
    st.set_page_config(page_title="Pomegranate Disease Detector", layout="wide")
    st.title("🌿 Pomegranate Disease Detection")
    
    model = load_model(MODEL_PATH, FILE_ID)
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner('Analyzing...'):
            predicted_label, confidence_score, all_probs = predict(image, model)

        if predicted_label == 'Healthy':
            st.success(f"Result: **{predicted_label}**")
        else:
            st.error(f"Result: **{predicted_label}**")
            
        st.metric("Confidence", f"{confidence_score:.2f}%")
        
        if all_probs:
            st.bar_chart(all_probs)

if __name__ == "__main__":
    main()