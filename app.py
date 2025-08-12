import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Define class labels (adjust to match your dataset)
class_labels = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
    "Squamous cell carcinoma",
    "Unknown"  # Replace or shorten as needed
]

# Load model architecture
model = models.mobilenet_v3(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(class_labels))  # 9 classes
model.load_state_dict(torch.load("mobilenet_mole.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

# Define prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Lesion Classifier",
    description="Upload an image of a mole or skin lesion to classify it. Powered by MobileNet and trained on ISIC dataset."
)

if __name__ == "__main__":
    interface.launch()
