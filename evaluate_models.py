import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import os


# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("skin_homeopathy_dataset.csv")

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["disease_name"])

classes = list(encoder.classes_)

print("\nNumber of classes:", len(classes))
print("Total images:", len(df))


# -------------------------
# Same Train-Test Split
# -------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Training images:", len(train_df))
print("Testing images:", len(test_df))


# -------------------------
# Dataset
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class SkinDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        return image, label


test_dataset = SkinDataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# -------------------------
# Accuracy Function
# -------------------------

def evaluate_model(model, model_name):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"{model_name} Accuracy: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

    return accuracy


# =========================================================
# CNN
# =========================================================

print("\n==============================")
print("Evaluating CNN")
print("==============================")


class CNNAgent(nn.Module):

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

            nn.Linear(
                128 * 28 * 28,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


cnn = CNNAgent(len(classes))

cnn.load_state_dict(
    torch.load(
        "cnn_agent.pth",
        map_location="cpu"
    )
)

cnn_accuracy = evaluate_model(
    cnn,
    "CNN"
)


# =========================================================
# EfficientNet
# =========================================================

print("\n==============================")
print("Evaluating EfficientNet")
print("==============================")


efficientnet = models.efficientnet_b0(
    weights=None
)

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    len(classes)
)

efficientnet.load_state_dict(
    torch.load(
        "efficientnet_agent.pth",
        map_location="cpu"
    )
)

efficientnet_accuracy = evaluate_model(
    efficientnet,
    "EfficientNet"
)


# =========================================================
# Swin Transformer
# =========================================================

print("\n==============================")
print("Evaluating Swin Transformer")
print("==============================")


swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=len(classes)
)

swin.load_state_dict(
    torch.load(
        "swin_agent.pth",
        map_location="cpu"
    )
)

swin_accuracy = evaluate_model(
    swin,
    "Swin Transformer"
)


# =========================================================
# Final Results
# =========================================================

print("\n===================================")
print("FINAL DEMO TEST ACCURACIES")
print("===================================")

print(f"CNN:          {cnn_accuracy:.2f}%")
print(f"EfficientNet: {efficientnet_accuracy:.2f}%")
print(f"Swin:         {swin_accuracy:.2f}%")

print("\nEvaluation completed.")