
import os
import json
import csv
import copy
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm


# ==========================================
# 1. CONFIGURATION
# ==========================================

TEST_DIR = "test"
BATCH_SIZE = 8
IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. SAME PREPROCESSING AS TRAINING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image_array = np.array(image)

    image_array = np.clip(image_array, 30, 220)

    image_array = (
        (image_array - 30) / (220 - 30)
    ) * 255

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    return image


test_transform = transforms.Compose([
    transforms.Lambda(preprocess_image),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

with open("class_names.json", "r") as f:
    trained_classes = json.load(f)

if test_dataset.classes != trained_classes:
    raise ValueError("Test class mapping does not match training!")

num_classes = len(trained_classes)

print("Number of classes:", num_classes)
print("Test images:", len(test_dataset))

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# 4. CNN ARCHITECTURE
# ==========================================

class CNN(nn.Module):

    def __init__(self):

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

        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, model_name):

    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    predictions = []

    print(f"\nEvaluating {model_name}...")

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            confidences, predicted = probabilities.max(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):

                actual_index = labels[i].item()
                predicted_index = predicted[i].item()

                class_total[actual_index] += 1

                if actual_index == predicted_index:
                    class_correct[actual_index] += 1

                predictions.append({
                    "Model": model_name,
                    "Actual": trained_classes[actual_index],
                    "Predicted": trained_classes[predicted_index],
                    "Confidence": round(
                        confidences[i].item() * 100, 2
                    ),
                    "Correct": actual_index == predicted_index
                })

    accuracy = (correct / total) * 100

    print(f"{model_name} accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

    return accuracy, predictions, class_correct, class_total


# ==========================================
# 6. LOAD CNN WEIGHTS
# ==========================================

cnn = CNN()

cnn_state = torch.load(
    "selection_container/cnn_agent.pth",
    map_location="cpu",
    weights_only=True
)

cnn.load_state_dict(cnn_state, strict=True)
cnn = cnn.to(DEVICE)

cnn_accuracy, cnn_predictions, cnn_correct, cnn_total = (
    evaluate_model(cnn, "CNN")
)

del cnn
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================
# 7. LOAD SWIN WEIGHTS
# ==========================================

swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=num_classes
)

swin_state = torch.load(
    "selection_container/swin_agent.pth",
    map_location="cpu",
    weights_only=True
)

swin.load_state_dict(swin_state, strict=True)
swin = swin.to(DEVICE)

swin_accuracy, swin_predictions, swin_correct, swin_total = (
    evaluate_model(swin, "Swin Transformer")
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

all_predictions = cnn_predictions + swin_predictions

with open("model_evaluation_results.csv", "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Actual",
            "Predicted",
            "Confidence",
            "Correct"
        ]
    )

    writer.writeheader()
    writer.writerows(all_predictions)


with open("model_class_accuracy.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Disease",
        "CNN Accuracy",
        "Swin Accuracy"
    ])

    for i, disease in enumerate(trained_classes):

        cnn_class_accuracy = (
            cnn_correct[i] / cnn_total[i] * 100
            if cnn_total[i] else 0
        )

        swin_class_accuracy = (
            swin_correct[i] / swin_total[i] * 100
            if swin_total[i] else 0
        )

        writer.writerow([
            disease,
            round(cnn_class_accuracy, 2),
            round(swin_class_accuracy, 2)
        ])


summary = {
    "test_images": len(test_dataset),
    "number_of_classes": num_classes,
    "device": str(DEVICE),
    "CNN_accuracy": round(cnn_accuracy, 2),
    "Swin_accuracy": round(swin_accuracy, 2)
}

with open("model_evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=4)


# ==========================================
# 9. FINAL RESULTS
# ==========================================

print("\n========== FINAL MODEL COMPARISON ==========")

print(f"CNN Accuracy: {cnn_accuracy:.2f}%")
print(f"Swin Accuracy: {swin_accuracy:.2f}%")
print("EfficientNet reported accuracy: 71.67%")

print("\nReports saved:")
print("1. model_evaluation_results.csv")
print("2. model_class_accuracy.csv")
print("3. model_evaluation_summary.json")

print("\nEvaluation completed!")
