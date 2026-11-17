import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

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

    # ---------------------------------
    # WINDOWING
    # ---------------------------------

    def apply_windowing(self,
                        image,
                        lower=30,
                        upper=220):

        image = np.array(image)

        image = np.clip(image, lower, upper)

        image = (
            (image - lower)
            / (upper - lower)
        ) * 255

        return Image.fromarray(
            image.astype(np.uint8)
        )

    # ---------------------------------
    # NOISE REMOVAL
    # ---------------------------------

    def remove_noise(self, image):

        return image.filter(
            ImageFilter.MedianFilter(size=3)
        )

    # ---------------------------------
    # COMPLETE PREPROCESSING PIPELINE
    # ---------------------------------

    def process(self, image_path):

        print("\nReal-time processing started...")

        image = Image.open(
            image_path
        ).convert("RGB")

        # Step 1: Windowing
        image = self.apply_windowing(image)

        # Step 2: Noise Removal
        image = self.remove_noise(image)

        # Step 3: Resize + Tensor + Normalize
        image = self.transform(image)

        print("Preprocessing completed.")

        return image.unsqueeze(0)
