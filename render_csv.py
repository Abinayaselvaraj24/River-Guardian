import csv
import os
import torch
from PIL import Image
from get_description import image_coordinates


class ImagePredictor:
    def __init__(self, images_directory, model_path, csv_file):
        self.images_directory = images_directory
        self.model =torch.hub.load('Utility_dir',
                           'custom', 'Artifacts/best.onnx', source='local')
        self.csv_file = csv_file

    def perform_inference(self):
        predictions = []

        for filename in os.listdir(self.images_directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image_path = os.path.join(self.images_directory, filename)
                img = Image.open(image_path).resize((640, 640))
                results = self.model(img)

                pred_labels = results.pandas().xyxy[0]['name'].tolist()
                pred_counts = pred_labels.count('plastic')

                url, lat, longi = image_coordinates(image_path)

                prediction = {
                    "IMG_ID": filename,
                    "PRED_LAB": "Yes" if pred_counts > 0 else "No",
                    "PRED_CT": pred_counts,
                    "GEO_Tag_URL": url,
                    "latitude": lat,
                    "longitude": longi
                }

                predictions.append(prediction)

        header = ["IMG_ID", "PRED_LAB", "PRED_CT", "GEO_Tag_URL", "latitude", "longitude"]

        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(predictions)

        print("CSV file created successfully.")