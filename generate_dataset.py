import os
import shutil
import random

source = "train"
destination = "demo_dataset/train"

os.makedirs(destination, exist_ok=True)

for disease in os.listdir(source):
    disease_path = os.path.join(source, disease)

    if os.path.isdir(disease_path):

        dest_folder = os.path.join(destination, disease)
        os.makedirs(dest_folder, exist_ok=True)

        images = [img for img in os.listdir(disease_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) > 0:
            selected = random.sample(images, min(2, len(images)))

            for img in selected:
                shutil.copy(
                    os.path.join(disease_path, img),
                    os.path.join(dest_folder, img)
                )

print("Demo dataset created successfully!")
