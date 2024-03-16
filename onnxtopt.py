import os
import xml.etree.ElementTree as ET


# Function to parse PASCAL VOC XML annotations and convert them to YOLO format
def convert_voc_to_yolo(xml_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue

        class_id = classes.index(class_name)

        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)

        # Convert coordinates to YOLO format (normalized)
        x_center = (x_min + x_max) / (2.0 * image_width)
        y_center = (y_min + y_max) / (2.0 * image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines


# Provide the path to your XML files and list of class names
xml_folder = "C:/Users/sanjai/Downloads/Label/Label"
class_names = ['plastic', 'weed']  # Add your class names here

output_folder = 'txtlabel'
os.makedirs(output_folder, exist_ok=True)

# Iterate through XML files and convert them to YOLO format
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)
        yolo_lines = convert_voc_to_yolo(xml_path, class_names)

        txt_filename = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, 'w') as txt_file:
            txt_file.write('\n'.join(yolo_lines))

print("Conversion completed.")