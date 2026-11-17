import os
import pandas as pd

# Path to demo dataset
dataset_path = "demo_dataset/train"

# Disease information dictionary
disease_info = {

    "Acne": {
        "symptoms": "Pimples, blackheads, whiteheads, oily skin",
        "remedies": "Sulphur, Kali Bromatum"
    },

    "Actinic_Keratosis": {
        "symptoms": "Rough scaly patches, dry skin lesions",
        "remedies": "Arsenicum Album, Thuja"
    },

    "Benign_tumors": {
        "symptoms": "Non-cancerous skin growths or lumps",
        "remedies": "Calcarea Fluorica, Thuja"
    },

    "Bullous": {
        "symptoms": "Blisters, fluid-filled lesions, itching",
        "remedies": "Cantharis, Rhus Toxicodendron"
    },

    "Candidiasis": {
        "symptoms": "Red rash, itching, white patches",
        "remedies": "Borax, Mercurius Solubilis"
    },

    "DrugEruption": {
        "symptoms": "Skin rash after medication, redness, itching",
        "remedies": "Apis Mellifica, Sulphur"
    },

    "Eczema": {
        "symptoms": "Dry skin, itching, redness, inflammation",
        "remedies": "Graphites, Petroleum"
    },

    "Infestations_Bites": {
        "symptoms": "Insect bites, itching, redness, swelling",
        "remedies": "Ledum Palustre, Apis Mellifica"
    },

    "Lichen": {
        "symptoms": "Purple itchy flat bumps, skin discoloration",
        "remedies": "Arsenicum Album, Sulphur"
    },

    "Lupus": {
        "symptoms": "Butterfly rash, fatigue, joint pain",
        "remedies": "Rhus Toxicodendron, Arsenicum Album"
    },

    "Moles": {
        "symptoms": "Dark skin spots or growths",
        "remedies": "Thuja, Calcarea Carbonica"
    },

    "Psoriasis": {
        "symptoms": "Scaly plaques, dry skin, itching",
        "remedies": "Arsenicum Album, Sepia"
    },

    "Rosacea": {
        "symptoms": "Facial redness, visible blood vessels",
        "remedies": "Sulphur, Belladonna"
    },

    "Seborrh_Keratoses": {
        "symptoms": "Waxy brown skin growths",
        "remedies": "Thuja, Calcarea Fluorica"
    },

    "SkinCancer": {
        "symptoms": "Abnormal skin growth, bleeding sores, changing moles",
        "remedies": "Consult dermatologist immediately"
    },

    "Sun_Sunlight_Damage": {
        "symptoms": "Sunburn, pigmentation, skin aging",
        "remedies": "Cantharis, Calendula"
    },

    "Tinea": {
        "symptoms": "Ring-shaped rash, itching, scaling",
        "remedies": "Tellurium, Sepia"
    },

    "Unknown_Normal": {
        "symptoms": "No visible skin disease",
        "remedies": "No treatment required"
    },

    "Vascular_Tumors": {
        "symptoms": "Red or purple vascular growths",
        "remedies": "Hamamelis, Calcarea Fluorica"
    },

    "Vasculitis": {
        "symptoms": "Purplish spots, ulcers, pain",
        "remedies": "Hamamelis, Lachesis"
    },

    "Vitiligo": {
        "symptoms": "White depigmented patches",
        "remedies": "Arsenicum Sulphuratum Flavum"
    },

    "Warts": {
        "symptoms": "Small rough skin growths",
        "remedies": "Thuja, Causticum"
    }
}

data = []

# Traverse disease folders
for disease in os.listdir(dataset_path):

    disease_path = os.path.join(dataset_path, disease)

    if os.path.isdir(disease_path):

        for img in os.listdir(disease_path):

            if img.lower().endswith(('.jpg', '.jpeg', '.png')):

                image_path = os.path.join(disease_path, img)

                symptoms = disease_info.get(
                    disease, {}
                ).get(
                    "symptoms",
                    "Symptoms not available"
                )

                remedies = disease_info.get(
                    disease, {}
                ).get(
                    "remedies",
                    "Remedy not available"
                )

                data.append([
                    image_path,
                    disease,
                    symptoms,
                    remedies
                ])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "image_path",
        "disease_name",
        "symptoms",
        "remedies"
    ]
)

# Save CSV
df.to_csv("skin_homeopathy_dataset.csv", index=False)

print("CSV created successfully!")
print("Total Images:", len(df))
print("Total Classes:", df["disease_name"].nunique())
