import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from data_processing_agent import DataProcessingAgent
# ---------------------------------
# MODEL SELECTED BY SELECTION AGENT
# ---------------------------------

selected_model = "efficientnet_agent.pth"   # Replace later with automatic selection

# ---------------------------------
# LOAD DATASET
# ---------------------------------

df = pd.read_csv("skin_homeopathy_dataset.csv")

classes = sorted(df["disease_name"].unique())

num_classes = len(classes)

print("Classes:", num_classes)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

if "cnn" in selected_model:

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

elif "efficientnet" in selected_model:

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

elif "swin" in selected_model:

    model = models.swin_t(weights=None)
    model.head = nn.Linear(
        model.head.in_features,
        num_classes
    )

else:
    print("Unknown model!")
    exit()

# ---------------------------------
# LOAD WEIGHTS
# ---------------------------------

model.load_state_dict(torch.load(selected_model, map_location="cpu"))
model.eval()

print(f"\nLoaded model: {selected_model}")

# ---------------------------------
# IMAGE TRANSFORM
# ---------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------
# INPUT IMAGE
# ---------------------------------

image_path = input("\nEnter image path: ").strip()

print("Path entered:", repr(image_path))
print("Exists:", os.path.exists(image_path))
if not os.path.exists(image_path):
    print("Image not found!")
    exit()

from data_processing_agent import DataProcessingAgent

processor = DataProcessingAgent()

image = processor.process(image_path)
# ---------------------------------
# PREDICTION
# ---------------------------------

with torch.no_grad():

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

predicted_disease = classes[predicted.item()]

print("\n===== PREDICTION RESULT =====")
print("Disease:", predicted_disease)

# ---------------------------------
# SYMPTOMS & REMEDIES
# ---------------------------------

result = df[df["disease_name"] == predicted_disease].iloc[0]

print("\nSymptoms:")
print(result["symptoms"])

print("\nHomeopathic Remedies:")
print(result["remedies"])
