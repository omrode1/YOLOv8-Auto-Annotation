import os
from ultralytics import YOLO
from PIL import Image
import argparse

def auto_annotate(model_path, image_folder, output_folder, conf_threshold):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            
            # Perform inference
            results = model(image_path, conf=conf_threshold)[0]
            
            # Create annotation file
            base_name = os.path.splitext(filename)[0]
            annotation_path = os.path.join(output_folder, f"{base_name}.txt")
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Write annotations
            with open(annotation_path, 'w') as f:
                for box in results.boxes:
                    class_id = int(box.cls)
                    x, y, w, h = box.xywhn[0]
                    
                    # Convert to YOLO format
                    f.write(f"{class_id} {x} {y} {w} {h}\n")
            
            print(f"Processed {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-annotate images using YOLOv8")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv8 model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to the output annotation folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    auto_annotate(args.model, args.input, args.output, args.conf)
