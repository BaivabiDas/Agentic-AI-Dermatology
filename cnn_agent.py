import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Load CSV
# -------------------------

df = pd.read_csv("skin_homeopathy_dataset.csv")

# Encode disease names
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['disease_name'])

# Save class names
classes = list(encoder.classes_)
print("Classes:", len(classes))

# Train-Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Dataset Class
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

        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        return image, label

train_dataset = SkinDataset(train_df)
test_dataset = SkinDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

# -------------------------
# CNN Model
# -------------------------

class CNNAgent(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNAgent(len(classes))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Training
# -------------------------

epochs = 10

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

# -------------------------
# MODEL EVALUATION
# -------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Total test samples:", total)
print("Correct predictions:", correct)
accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), "cnn_agent.pth")

print("CNN Agent Saved Successfully!")
