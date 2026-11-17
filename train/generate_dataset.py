import os
import csv

dataset_path = "."

with open("skin_homeopathy_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image_path",
        "disease",
        "symptoms",
        "homeopathic_remedies"
    ])

    for disease in os.listdir(dataset_path):

        if not os.path.isdir(disease):
            continue

        symptoms = "Refer RAG knowledge base"
        remedies = "Refer RAG knowledge base"

        for image in os.listdir(disease):

            writer.writerow([
                os.path.join(disease, image),
                disease,
                symptoms,
                remedies
            ])

print("CSV created successfully")
