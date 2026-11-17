import os
import psutil
import shutil

# --------------------------------
# MODEL INFORMATION
# --------------------------------

models = {
    "cnn_agent.pth": {
        "accuracy": 0.0,   # Replace with real accuracy later
        "type": "CNN"
    },

    "efficientnet_agent.pth": {
        "accuracy": 0.0,   # Replace with real accuracy later
        "type": "EfficientNet"
    },

    "swin_agent.pth": {
        "accuracy": 0.0,   # Replace with real accuracy later
        "type": "Swin Transformer"
    }
}

# --------------------------------
# SYSTEM STATUS
# --------------------------------

available_ram = psutil.virtual_memory().available / (1024 ** 3)
cpu_usage = psutil.cpu_percent(interval=1)

print("\n===== SYSTEM STATUS =====")
print(f"Available RAM : {available_ram:.2f} GB")
print(f"CPU Usage     : {cpu_usage}%")

# --------------------------------
# CHECK AVAILABLE MODELS
# --------------------------------

available_models = []

for model_name, info in models.items():

    if os.path.exists(model_name):

        model_size = os.path.getsize(model_name) / (1024 ** 2)

        score = (
            info["accuracy"]
            - (model_size * 0.2)
            + (available_ram * 2)
            - (cpu_usage * 0.1)
        )

        available_models.append({
            "name": model_name,
            "type": info["type"],
            "accuracy": info["accuracy"],
            "size": model_size,
            "score": score
        })

# --------------------------------
# NO MODELS FOUND
# --------------------------------

if not available_models:
    print("\nNo trained models found!")
    exit()

# --------------------------------
# RESOURCE FILTERING
# --------------------------------

suitable_models = []

for model in available_models:

    if model["type"] == "Swin Transformer":

        if available_ram < 4 or cpu_usage > 80:
            continue

    elif model["type"] == "EfficientNet":

        if available_ram < 2 or cpu_usage > 90:
            continue

    suitable_models.append(model)

# Fallback if none satisfy constraints

if not suitable_models:
    suitable_models = available_models

# --------------------------------
# SELECT BEST MODEL
# --------------------------------

best_model = max(
    suitable_models,
    key=lambda x: x["score"]
)

with open("/shared/selected_model.txt", "w") as f:
    f.write(best_model["name"])

# Copy the selected model to the shared volume
shutil.copy(
    best_model["name"],
    "/shared/selected_model.pth"
)

print(f"Selected model copied to shared volume: {best_model['name']}")

# --------------------------------
# DISPLAY DETAILS
# --------------------------------

print("\n===== AVAILABLE MODELS =====")

for model in available_models:

    print(
        f"\nModel: {model['name']}"
    )
    print(f"Type      : {model['type']}")
    print(f"Accuracy  : {model['accuracy']}%")
    print(f"Size      : {model['size']:.2f} MB")
    print(f"Score     : {model['score']:.2f}")

print("\n===== SELECTED MODEL =====")
print(f"Model Name : {best_model['name']}")
print(f"Type       : {best_model['type']}")
print(f"Accuracy   : {best_model['accuracy']}%")
print(f"Size       : {best_model['size']:.2f} MB")

selected_model = best_model["name"]

print("\nSelection Agent completed successfully!")
