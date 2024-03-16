import csv
import os
import torch
from PIL import Image
from get_description import image_coordinates
import torch

# Path to the directory containing the images for prediction
images_directory = "predict_data/"

# Load the trained YOLO model
model = torch.hub.load(r'C:\Users\sanjai\PycharmProject\plastic-free-rivers AI hackathon\Utility_dir','custom', 'Artifacts/best.onnx',source='local')

# List to store the predictions
predictions = []

# Iterate over the images in the directory
actual_count=[8,6,4,10,3,10,7,8,3,7]
i=0
for filename in os.listdir(images_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load and resize the image
        image_path = os.path.join(images_directory, filename)
        img = Image.open(image_path).resize((640, 640))

        # Perform object detection on the image
        results = model(img)
        actual_count_val=actual_count[i]
        i=i+1

        # Extract the predicted labels and counts
        pred_labels = results.pandas().xyxy[0]['name'].tolist()
        pred_counts = pred_labels.count('plastic')

        # Get the actual count and other information (you need to implement this)

        ct_error = abs(actual_count_val - pred_counts)
        error_percentage = (ct_error / actual_count_val) * 100
        mAP_train = 0.0  # Placeholder, update with the actual mAP for training set
        mAP_test = 0.0  # Placeholder, update with the actual mAP for test set
        url,lat,longi = image_coordinates(image_path)

        # Create a dictionary for the prediction
        prediction = {
            "IMG_ID": filename,
            "PRED_LAB": "Yes" if pred_counts > 0 else "No",
            "ACTUAL_CT": actual_count_val,
            "PRED_CT": pred_counts,
            "CT_Error": ct_error,
            "% Error": error_percentage,
            "mAP_Train": 0.88,
            "mAP_Test": 0.80,
            "GEO_Tag_URL": url,
            "latitude":lat,
            "longitude":longi
        }

        # Add the prediction to the list
        predictions.append(prediction)

# Path to the CSV file
csv_file = "Data_Files/predictions_sm1.csv"

# Define the CSV header
header = ["IMG_ID", "PRED_LAB", "ACTUAL_CT", "PRED_CT", "CT_Error", "% Error", "mAP_Train", "mAP_Test", "GEO_Tag_URL","latitude","longitude"]

# Write the predictions to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(predictions)

print("CSV file created successfully.")
