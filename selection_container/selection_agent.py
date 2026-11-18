
import os
import psutil
import shutil
import json
import time

start_time = time.perf_counter()

# --------------------------------
# MODEL INFORMATION
# --------------------------------

models = {
    "cnn_agent.pth": {
        "accuracy": 40.75,
        "type": "CNN"
    },

    "efficientnet_agent.pth": {
        "accuracy": 71.67,
        "type": "EfficientNet"
    },

    "swin_agent.pth": {
        "accuracy": 78.78,
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

    else:
        print(f"\nModel file missing: {model_name}")

# --------------------------------
# NO MODELS FOUND
# --------------------------------

if not available_models:
    raise FileNotFoundError(
        "No trained model files found in Selection Agent!"
    )

# --------------------------------
# RESOURCE FILTERING
# --------------------------------

suitable_models = []

for model in available_models:

    if model["type"] == "CNN":

        if available_ram < 2 or cpu_usage > 90:
            continue

    elif model["type"] == "EfficientNet":

        if available_ram < 2 or cpu_usage > 90:
            continue

    elif model["type"] == "Swin Transformer":

        if available_ram < 4 or cpu_usage > 80:
            continue

    suitable_models.append(model)

# --------------------------------
# NO SUITABLE MODEL
# --------------------------------

if not suitable_models:
    raise RuntimeError(
        "No model satisfies the current resource requirements. "
        "Selection stopped instead of choosing an unsuitable model."
    )

# --------------------------------
# SELECT BEST MODEL
# --------------------------------

best_model = max(
    suitable_models,
    key=lambda x: x["score"]
)

# --------------------------------
# SAVE SELECTED MODEL
# --------------------------------

os.makedirs("/shared", exist_ok=True)

with open("/shared/selected_model.txt", "w") as f:
    f.write(best_model["name"])

shutil.copy2(
    best_model["name"],
    "/shared/selected_model.pth"
)

with open("selected_model.txt", "w") as f:
    f.write(best_model["name"])

print(
    f"\nSelected model copied to shared volume: "
    f"{best_model['name']}"
)

# --------------------------------
# DISPLAY AVAILABLE MODELS
# --------------------------------

print("\n===== AVAILABLE MODELS =====")

for model in available_models:

    print(f"\nModel    : {model['name']}")
    print(f"Type     : {model['type']}")
    print(f"Accuracy : {model['accuracy']}%")
    print(f"Size     : {model['size']:.2f} MB")
    print(f"Score    : {model['score']:.2f}")
    print(
        f"Resource eligible: "
        f"{model in suitable_models}"
    )

# --------------------------------
# DISPLAY SELECTED MODEL
# --------------------------------

print("\n===== SELECTED MODEL =====")
print(f"Model Name : {best_model['name']}")
print(f"Type       : {best_model['type']}")
print(f"Accuracy   : {best_model['accuracy']}%")
print(f"Size       : {best_model['size']:.2f} MB")
print(f"Score      : {best_model['score']:.2f}")

# --------------------------------
# SAVE SELECTION METADATA
# --------------------------------

selection_time = time.perf_counter() - start_time

selection_info = {
    "selected_model": best_model["name"],
    "model_type": best_model["type"],
    "accuracy": best_model["accuracy"],
    "model_size_mb": round(best_model["size"], 2),
    "selection_score": round(best_model["score"], 2),
    "available_ram_gb": round(available_ram, 2),
    "cpu_usage_percent": cpu_usage,
    "selection_time_seconds": round(selection_time, 4)
}

with open("/shared/selection_info.json", "w") as f:
    json.dump(selection_info, f, indent=4)

print("\nSelection metadata saved to shared volume!")
print("\nSelection Agent completed successfully!")
