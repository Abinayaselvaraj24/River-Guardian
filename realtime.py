import torch
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import streamlit as st
def change_font_color(element, text, font_color):
    st.markdown(f'<{element} style="color: {font_color};">{text}</{element}>', unsafe_allow_html=True)

def process_frame(frame):
    # Convert OpenCV frame to PIL image
    model = torch.hub.load('Utility_dir',
                           'custom', 'Artifacts/best.onnx', source='local')
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 640))

    # Perform object detection on the image
    results = model(img, size=640)

    # Get the predicted labels and their counts
    pred_labels = results.pandas().xyxy[0]['name']
    plastic_count = (pred_labels == 'plastic').sum()

    # Filter the results to only include "plastic" class
    plastic_boxes = results.pred[0][pred_labels == 'plastic']
    df = results.pandas().xyxy[0]
    plastic_scores = list(df[df['name'] == 'plastic']['confidence'])

    # Draw bounding boxes and labels for the filtered "plastic" class
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 14)

    for box, score in zip(plastic_boxes, plastic_scores):
        x1, y1, x2, y2 = box[0:4]
        coordinates = [(x1, y1), (x2, y2)]

        draw.rectangle(coordinates, outline=(0, 255, 0), width=2)

        label = f"Plastic: {score:.2f}"
        label_font = ImageFont.truetype("arial.ttf", 12)
        label_size = label_font.getsize(label)
        draw.rectangle([(x1, y1), (x1 + label_size[0] + 4, y1 + label_size[1] + 4)], fill=(0, 255, 0))
        draw.text((x1 + 2, y1 + 2), label, fill=(0, 0, 0), font=label_font)

    return np.array(img)

def realtime_input():
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)  # Assuming you have this function defined

        # Display the processed frame with bounding boxes
        stframe.image(processed_frame, channels="RGB")

    cap.release()


# Call the function with appropriate arguments
def realtime():
    change_font_color("h1","Plastic Detection in Video","black")
    selected_option = st.sidebar.radio("Select input source", ["Real-time camera", "Upload video"])

    vid_file = None
    if selected_option == "Real-time camera":
        realtime_input()
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "Data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)

            # Display the processed frame with bounding boxes
            stframe.image(processed_frame, channels="RGB")

        cap.release()

    else:
        change_font_color('p',"Upload a Video",'black')






