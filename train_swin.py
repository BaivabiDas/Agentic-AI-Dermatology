
import os
import json
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import timm

from PIL import Image, ImageFilter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_DIR = "train"
TEST_DIR = "test"

BATCH_SIZE = 8
EPOCHS = 20
IMAGE_SIZE = 224
LEARNING_RATE = 0.00003
PATIENCE = 5
VALIDATION_SPLIT = 0.10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== SWIN TRAINING CONFIGURATION =====")
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


image_transform = transforms.Compose([
    transforms.Lambda(preprocess_image),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD DATASETS
# ==========================================

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError("Train folder not found!")

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError("Test folder not found!")

full_train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=image_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=image_transform
)

classes = full_train_dataset.classes
num_classes = len(classes)

if classes != test_dataset.classes:
    raise ValueError("Train and test classes do not match!")

validation_size = int(
    len(full_train_dataset) * VALIDATION_SPLIT
)

training_size = (
    len(full_train_dataset) - validation_size
)

generator = torch.Generator().manual_seed(SEED)

train_subset, validation_subset = random_split(
    full_train_dataset,
    [training_size, validation_size],
    generator=generator
)

with open("class_names.json", "w") as f:
    json.dump(classes, f, indent=4)

print("\n===== DATASET INFORMATION =====")
print("Training images:", training_size)
print("Validation images:", validation_size)
print("Testing images:", len(test_dataset))
print("Number of classes:", num_classes)


# ==========================================
# 4. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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
# 5. LOAD PRETRAINED SWIN-TINY
# ==========================================

print("\nLoading pretrained Swin-Tiny...")

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=num_classes
)

model = model.to(DEVICE)

print("Swin-Tiny loaded successfully!")


# ==========================================
# 6. TRAINING SETUP
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.05
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.inference_mode():

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

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = running_loss / len(loader)

    return average_loss, accuracy


# ==========================================
# 8. TRAINING LOOP
# ==========================================

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

print("\n===== SWIN TRAINING STARTED =====")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

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

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)}"
            )

    train_loss = running_loss / len(train_loader)

    val_loss, val_accuracy = evaluate(
        model,
        validation_loader
    )

    scheduler.step()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_weights,
            "swin_agent.pth"
        )

        epochs_without_improvement = 0

        print("Best Swin model saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# ==========================================
# 9. FINAL TEST EVALUATION
# ==========================================

model.load_state_dict(best_weights)

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print("\n===== FINAL SWIN TEST RESULTS =====")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

results = {
    "model": "Swin-Tiny",
    "number_of_classes": num_classes,
    "training_images": training_size,
    "validation_images": validation_size,
    "testing_images": len(test_dataset),
    "test_accuracy": round(test_accuracy, 2),
    "best_validation_accuracy": round(best_accuracy, 2),
    "epochs_completed": epoch + 1,
    "device": str(DEVICE)
}

with open("swin_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSwin training completed!")
print("Model: swin_agent.pth")
print("Metrics: swin_metrics.json")
