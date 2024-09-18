# YOLOv8 Auto-Annotation Tool

This project provides a Python script for automatically annotating images using a YOLOv8 model. It processes a folder of images, generates bounding box annotations, and saves them in YOLO format.

## Features

- Auto-annotate images using a pre-trained YOLOv8 model
- Adjustable confidence threshold for object detection
- Batch processing of multiple images
- Output in YOLO format (class_id, x_center, y_center, width, height)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/yolov8-auto-annotate.git
   cd yolov8-auto-annotate
   ```

2. Install the required packages:
   ```
   pip install ultralytics pillow
   ```

## Usage

### From Command Line
To use the YOLOv8 Auto-Annotation tool, follow these steps:

1. Prepare your YOLOv8 model (.pt file)
2. Organize your images in a single folder
3. Run the script with the following command:

```
python auto_annotate.py --model path/to/your/model.pt --input path/to/image/folder --output path/to/output/folder --conf 0.3
```
### from tkinter GUI
To use the YOLOv8 Auto-Annotation tool from tkinter , follow these steps:

1. Prepare your YOLOv8 model (.pt file)
2. Organize your images in a single folder
3. Run the script with the following command:

```
python auto_annotate_tkinter.py
```


### Command-line Arguments

- `--model`: Path to the YOLOv8 model file (required)
- `--input`: Path to the folder containing images to annotate (required)
- `--output`: Path to save the annotation files (required)
- `--conf`: Confidence threshold for object detection (default: 0.25)

## Output

The script will create a text file for each image in the specified output folder. Each text file will contain annotations in the YOLO format:

```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer representing the class of the detected object
- `center_x`, `center_y`: Coordinates of the center of the bounding box (normalized)
- `width`, `height`: Width and height of the bounding box (normalized)

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) implementation.

## Contact

If you have any questions or feedback, please open an issue on this GitHub repository.
