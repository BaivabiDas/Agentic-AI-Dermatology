import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import json
import time
from flask import Flask, request, jsonify
from data_processing_agent import DataProcessingAgent
import urllib.request

app = Flask(__name__)
# ---------------------------------
# LOAD SELECTION METADATA
# ---------------------------------

selection_info_file = "/shared/selection_info.json"

with open(selection_info_file, "r") as f:
    selection_info = json.load(f)
# ---------------------------------
# MODEL SELECTED BY SELECTION AGENT
# ---------------------------------
selected_model_file = "/shared/selected_model.txt"
if not os.path.exists(selected_model_file):
    raise FileNotFoundError(
        "Selection Agent did not provide selected_model.txt"
    )

with open(selected_model_file, "r") as f:
    selected_model = f.read().strip()

print(f"\nModel selected by Selection Agent: {selected_model}")

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

# Load the model selected by the Selection Agent
model_path = "/shared/selected_model.pth"

model.load_state_dict(
    torch.load(model_path, map_location="cpu")
)
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
# PREDICTION API
# ---------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Prediction Agent is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    total_start = time.perf_counter()

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No image selected"
        }), 400

    image_path = os.path.join("/tmp", file.filename)

    try:

        file.save(image_path)

        # ---------------------------------
        # DATA PROCESSING
        # ---------------------------------

        processor = DataProcessingAgent()
        image = processor.process(image_path)

        # ---------------------------------
        # PREDICTION
        # ---------------------------------
        prediction_start = time.perf_counter()

        with torch.no_grad():

            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        prediction_time = time.perf_counter() - prediction_start
        predicted_disease = classes[predicted.item()]
        # ---------------------------------
        # SYMPTOMS & REMEDIES
        # ---------------------------------

        result = df[
            df["disease_name"] == predicted_disease
        ].iloc[0]

        symptoms = result["symptoms"]
        remedies = result["remedies"]

        print("\nSymptoms:")
        print(symptoms)

        print("\nHomeopathic Remedies:")
        print(remedies)

        # ---------------------------------
        # AI RECOMMENDATION
        # ---------------------------------

        

        print("\n===== AI RECOMMENDATION =====")

        recommendation_url = "http://recommendation-agent:5002/recommend"

        payload = json.dumps({
            "disease": predicted_disease
        }).encode("utf-8")

        req = urllib.request.Request(
            recommendation_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            recommendation_data = json.loads(
                response.read().decode("utf-8")
            )

        recommendation = recommendation_data["recommendation"]

        print(recommendation)

        # ---------------------------------
        # RETURN RESULT
        # ---------------------------------

        total_time = time.perf_counter() - total_start

        return jsonify({
            "status": "success",

            "disease": predicted_disease,
            "confidence_percent": round(confidence.item() * 100, 2),

            "selected_model": selection_info["selected_model"],
            "model_type": selection_info["model_type"],
            "model_accuracy": selection_info["accuracy"],
            "model_size_mb": selection_info["model_size_mb"],
            "selection_score": selection_info["selection_score"],

            "available_ram_gb": selection_info["available_ram_gb"],
            "cpu_usage_percent": selection_info["cpu_usage_percent"],
            "selection_time_seconds": selection_info["selection_time_seconds"],

            "prediction_time_seconds": round(prediction_time, 4),
            "total_execution_time_seconds": round(total_time, 4),

            "symptoms": symptoms,
            "remedies": remedies,
            "recommendation": recommendation
        })
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
# ---------------------------------
# START FLASK SERVER
# ---------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )