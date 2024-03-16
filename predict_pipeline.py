import argparse
import os
import torch
from get_description import image_coordinates
from PIL import Image, ImageDraw, ImageFont

def predict_image(image_path, output_dir):
    # Load the trained model
    url, lat, longi = image_coordinates(image_path)
    model = torch.hub.load('Utility_dir',
                           'custom', 'Artifacts/best.onnx', source='local')

    # Load and preprocess the image
    img = Image.open(image_path).resize((1280, 1280))

    # Perform object detection on the image
    results = model(img, size=640)

    # Get the predicted labels and their counts
    pred_labels = results.pandas().xyxy[0]['name']
    print(pred_labels)
    plastic_count = (pred_labels == 'plastic').sum()

    # Filter the results to only include "plastic" class
    plastic_boxes = results.pred[0][pred_labels == 'plastic']
    # get the plastic confidense score
    df = results.pandas().xyxy[0]
    print(df)

    plastic_scores = list(df[df['name'] == 'plastic']['confidence'])
    print(plastic_scores)

    # Draw bounding boxes and labels for the filtered "plastic" class
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 14)  # Specify the font and size for the label

    for box, score in zip(plastic_boxes, plastic_scores):
        # Extract coordinates
        x1, y1, x2, y2 = box[0:4]
        coordinates = [(x1, y1), (x2, y2)]

        # Draw rectangle
        draw.rectangle(coordinates, outline=(0, 255, 0), width=2)

        # Add label "Plastic" with confidence score
        label = f"Plastic: {score:.2f}"
        label_font = ImageFont.truetype("arial.ttf", 12)  # Specify font and size for the label text
        label_size = label_font.getsize(label)
        draw.rectangle([(x1, y1), (x1 + label_size[0] + 4, y1 + label_size[1] + 4)], fill=(0, 255, 0))
        draw.text((x1 + 2, y1 + 2), label, fill=(0, 0, 0), font=label_font)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)




    # Save the predicted image with bounding boxes
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    img.save(save_path)

    # Get the GeoTag URL


    return plastic_count, url, save_path,lat,longi




