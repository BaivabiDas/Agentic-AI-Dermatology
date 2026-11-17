import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# -------------------------
# Load CSV
# -------------------------

df = pd.read_csv("skin_homeopathy_dataset.csv")

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['disease_name'])

classes = list(encoder.classes_)
print("Classes:", len(classes))

# -------------------------
# Train Test Split
# -------------------------

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
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
)

# -------------------------
# Swin Transformer Model
# -------------------------

model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=len(classes)
)

# -------------------------
# Training Setup
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# -------------------------
# Training Loop
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
# Save Model
# -------------------------

torch.save(
    model.state_dict(),
    "swin_agent.pth"
)

print("Swin Agent Saved Successfully!")
