import torch
import torch.nn as nn
from torchvision import models
import timm
import pandas as pd
import os
import json
import time
import uuid
from flask import Flask, request, jsonify
from data_processing_agent import DataProcessingAgent
import urllib.request

app = Flask(__name__)

# ---------------------------------
# LOAD SELECTION METADATA
# ---------------------------------

selection_info_file = "/shared/selection_info.json"

if not os.path.exists(selection_info_file):
    raise FileNotFoundError(
        "Selection Agent did not provide selection_info.json"
    )

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

print("Number of classes:", num_classes)

# ---------------------------------
# CNN ARCHITECTURE
# ---------------------------------

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------
# INITIALIZE SELECTED MODEL
# ---------------------------------

if "cnn" in selected_model.lower():

    model = CNN(num_classes)

elif "efficientnet" in selected_model.lower():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

elif "swin" in selected_model.lower():

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes
    )

else:
    raise ValueError(
        f"Unknown model selected: {selected_model}"
    )

# ---------------------------------
# LOAD TRAINED MODEL WEIGHTS
# ---------------------------------

model_path = "/shared/selected_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Selected model weights were not found!"
    )

state_dict = torch.load(
    model_path,
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()

print(f"\nSuccessfully loaded model: {selected_model}")

# ---------------------------------
# IMAGE PROCESSING AGENT
# ---------------------------------

processor = DataProcessingAgent()

# ---------------------------------
# HEALTH API
# ---------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "Prediction Agent is running",
        "selected_model": selected_model,
        "model_type": selection_info.get("model_type")
    })


# ---------------------------------
# PREDICTION API
# ---------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    total_start = time.perf_counter()

    image_path = None

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

    try:

        # ---------------------------------
        # SAVE UPLOADED IMAGE
        # ---------------------------------

        image_path = os.path.join(
            "/tmp",
            f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
        )

        file.save(image_path)

        # ---------------------------------
        # DATA PROCESSING
        # ---------------------------------

        image = processor.process(image_path)

        # ---------------------------------
        # MODEL PREDICTION
        # ---------------------------------

        prediction_start = time.perf_counter()

        with torch.no_grad():

            outputs = model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        prediction_time = (
            time.perf_counter() - prediction_start
        )

        predicted_disease = classes[predicted.item()]

        confidence_percent = round(
            confidence.item() * 100,
            2
        )

        print("\n===== PREDICTION RESULT =====")
        print("Disease:", predicted_disease)
        print("Confidence:", confidence_percent, "%")

        # ---------------------------------
        # SYMPTOMS AND REMEDIES
        # ---------------------------------

        result = df[
            df["disease_name"] == predicted_disease
        ].iloc[0]

        symptoms = result["symptoms"]
        remedies = result["remedies"]

        # ---------------------------------
        # AI RECOMMENDATION AGENT
        # ---------------------------------

        print("\n===== AI RECOMMENDATION =====")

        recommendation_url = (
            "http://recommendation-agent:5002/recommend"
        )

        payload = json.dumps({
            "disease": predicted_disease
        }).encode("utf-8")

        req = urllib.request.Request(
            recommendation_url,
            data=payload,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            req,
            timeout=120
        ) as response:

            recommendation_data = json.loads(
                response.read().decode("utf-8")
            )

        recommendation = recommendation_data["recommendation"]

        print(recommendation)

        # ---------------------------------
        # EXECUTION TIME
        # ---------------------------------

        total_time = (
            time.perf_counter() - total_start
        )

        # ---------------------------------
        # RETURN PREDICTION RESULT
        # ---------------------------------

        return jsonify({

            "status": "success",

            "disease": predicted_disease,
            "confidence_percent": confidence_percent,

            "selected_model": selection_info.get(
                "selected_model", selected_model
            ),

            "model_type": selection_info.get(
                "model_type"
            ),

            "model_accuracy": selection_info.get(
                "accuracy"
            ),

            "model_size_mb": selection_info.get(
                "model_size_mb",
                selection_info.get("size")
            ),

            "selection_score": selection_info.get(
                "selection_score"
            ),

            "available_ram_gb": selection_info.get(
                "available_ram_gb"
            ),

            "cpu_usage_percent": selection_info.get(
                "cpu_usage_percent"
            ),

            "selection_time_seconds": selection_info.get(
                "selection_time_seconds"
            ),

            "prediction_time_seconds": round(
                prediction_time, 4
            ),

            "total_execution_time_seconds": round(
                total_time, 4
            ),

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

        if image_path and os.path.exists(image_path):
            os.remove(image_path)


# ---------------------------------
# START FLASK SERVER
# ---------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
