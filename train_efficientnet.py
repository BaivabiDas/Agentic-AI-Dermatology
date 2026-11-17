
import os
import json
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from PIL import Image, ImageFilter


# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_DIR = "train"
TEST_DIR = "test"

BATCH_SIZE = 16
EPOCHS = 5
IMAGE_SIZE = 224
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== TRAINING CONFIGURATION =====")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3), 2
        ),
        "GB"
    )


# ==========================================
# 2. IMAGE PREPROCESSING
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


train_transform = transforms.Compose([
    transforms.Lambda(preprocess_image),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
# 3. LOAD DATASET
# ==========================================

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError("Train folder not found!")

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError("Test folder not found!")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

classes = train_dataset.classes

if classes != test_dataset.classes:
    raise ValueError(
        "Train and test class folders do not match!"
    )

num_classes = len(classes)

print("\n===== DATASET INFORMATION =====")
print("Number of classes:", num_classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

print("\nDisease Classes:")

for disease in classes:
    print(disease)

# Save exact class mapping
with open("class_names.json", "w") as f:
    json.dump(classes, f, indent=4)


# ==========================================
# 4. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ==========================================
# 5. LOAD PRETRAINED EFFICIENTNET-B0
# ==========================================

print("\nLoading pretrained EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)


# ==========================================
# 6. TRAINING SETUP
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ==========================================
# 7. VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    average_loss = running_loss / len(loader)

    return average_loss, accuracy


# ==========================================
# 8. TRAINING LOOP
# ==========================================

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

print("\n===== TRAINING STARTED =====")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_loss, val_accuracy = evaluate(
        model,
        test_loader
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Evaluation Loss: {val_loss:.4f}")
    print(f"Evaluation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_weights,
            "efficientnet_agent.pth"
        )

        print("Best model saved!")


# ==========================================
# 9. FINAL TEST EVALUATION
# ==========================================

model.load_state_dict(best_weights)

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print("\n===== FINAL TEST RESULTS =====")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save actual measured results
results = {
    "model": "EfficientNet-B0",
    "number_of_classes": num_classes,
    "training_images": len(train_dataset),
    "testing_images": len(test_dataset),
    "test_accuracy": round(test_accuracy, 2),
    "best_evaluation_accuracy": round(best_accuracy, 2),
    "epochs": EPOCHS,
    "device": str(DEVICE)
}

with open("efficientnet_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nEfficientNet training completed!")
print("Model: efficientnet_agent.pth")
print("Class mapping: class_names.json")
print("Metrics: efficientnet_metrics.json")
