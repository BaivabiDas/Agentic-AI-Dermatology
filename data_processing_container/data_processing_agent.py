import base64
import io

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from flask import Flask, request, jsonify


class DataProcessingAgent:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # -------------------------------
    # WINDOWING
    # -------------------------------

    def apply_windowing(self, image, lower=30, upper=220):

        image = np.array(image)

        image = np.clip(image, lower, upper)

        image = (
            (image - lower)
            / (upper - lower)
        ) * 255

        return Image.fromarray(
            image.astype(np.uint8)
        )

    # -------------------------------
    # NOISE REMOVAL
    # -------------------------------

    def remove_noise(self, image):

        return image.filter(
            ImageFilter.MedianFilter(size=3)
        )

    # -------------------------------
    # COMPLETE PREPROCESSING
    # -------------------------------

    def process(self, image):

        print("\nReal-time processing started...")

        # Step 1: Windowing
        image = self.apply_windowing(image)

        # Step 2: Noise Removal
        image = self.remove_noise(image)

        # Step 3: Resize + Tensor + Normalize
        image = self.transform(image)

        print("Preprocessing completed.")

        return image


# ---------------------------------------
# FLASK API
# ---------------------------------------

app = Flask(__name__)

processor = DataProcessingAgent()


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "Data Processing Agent is running"
    })


@app.route("/process", methods=["POST"])
def process_image():

    try:

        # Check whether image was received
        if "image" not in request.files:

            return jsonify({
                "error": "No image file received"
            }), 400

        file = request.files["image"]

        # Open image
        image = Image.open(file.stream).convert("RGB")

        # Process image
        tensor = processor.process(image)

        # Convert tensor to numpy
        array = tensor.numpy().astype(np.float32)

        # Convert numpy array to bytes
        buffer = io.BytesIO()

        np.save(buffer, array)

        # Encode bytes to Base64
        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({

            "status": "success",

            "shape": list(array.shape),

            "data": encoded

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ---------------------------------------
# START SERVER
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )