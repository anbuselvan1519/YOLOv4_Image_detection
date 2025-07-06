🧠 YOLOv4 Object Detection using OpenCV
========================================

This project demonstrates object detection on static images using the YOLOv4 model with OpenCV’s DNN module. YOLO (You Only Look Once) is one of the fastest and most accurate real-time object detection algorithms.

The code loads the YOLOv4 model, performs object detection on an image, and displays the result with bounding boxes and confidence scores for detected objects.

--------------------------------------------------------------------------------
Demo Output
--------------------------------------------------------------------------------

Example input image: Canal-Street.jpg

After running the detection, you will see bounding boxes like:

person: 0.91
car: 0.88
traffic light: 0.82

A window will pop up showing the detection result.

--------------------------------------------------------------------------------
📂 Project Structure
--------------------------------------------------------------------------------

yolov4-object-detection/
├── yolo_detection_image.py               -> Main object detection script
├── yolov4.cfg              -> YOLOv4 model configuration
├── yolov4_weights          -> Pretrained YOLOv4 weights (download separately)
├── coco.names              -> Class labels for COCO dataset
├── Sample_image.jpg        -> Sample input image
├── Output_image.jpg        -> Sample output image
├── requirements.txt        -> Required Python packages
└── README.md               -> Project documentation

--------------------------------------------------------------------------------
🔧 Installation
--------------------------------------------------------------------------------

1. Clone this repository:
   git clone https://github.com/anbuselvan1519/YOLOv4_Image_detection.git
   cd yolov4-object-detection

2. Install the required Python packages:
   pip install -r requirements.txt

--------------------------------------------------------------------------------
Download Required Files
--------------------------------------------------------------------------------

You need to download the following files manually and place them in the project root directory:

- yolov4.weights:
  https://pjreddie.com/media/files/yolov4.weights

- yolov4.cfg:
  https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg

- coco.names:
  https://github.com/pjreddie/darknet/blob/master/data/coco.names

--------------------------------------------------------------------------------
How to Run
--------------------------------------------------------------------------------

To perform object detection on the sample image:

   python yolo_detection_image.py

To run it on a different image, modify the last line in detect.py:

   detect_objects_in_image("your_image.jpg")

--------------------------------------------------------------------------------
How It Works
--------------------------------------------------------------------------------

- Loads YOLOv4 weights and configuration using OpenCV's DNN module
- Loads class labels from coco.names
- Reads and preprocesses the input image (416x416)
- Converts the image to a blob and feeds it to the network
- Extracts outputs from the output layers
- Applies Non-Maximum Suppression (NMS)
- Draws bounding boxes with labels and confidence scores

--------------------------------------------------------------------------------
Customization
--------------------------------------------------------------------------------

- Change confidence threshold:
  In detect.py, change:
     if confidence > 0.8:

- Change input resolution:
  Modify (416, 416) inside cv2.dnn.blobFromImage()

- Detect multiple images:
  Wrap the function in a loop with different image paths

--------------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------------

These are included in requirements.txt:

   opencv-python>=4.5.1
   numpy>=1.19.5

Install using:
   pip install -r requirements.txt

--------------------------------------------------------------------------------
Limitations
--------------------------------------------------------------------------------

- Detects only in static images
- Requires manual download of YOLO files
- No webcam or video support
- GPU acceleration not included (requires OpenCV with CUDA)

--------------------------------------------------------------------------------
License
--------------------------------------------------------------------------------

This project is licensed under the MIT License.

YOLOv4 model and weights are provided by:
- Joseph Redmon (https://pjreddie.com/darknet/yolo/)
- Alexey Bochkovskiy (https://github.com/AlexeyAB)

--------------------------------------------------------------------------------
Author
--------------------------------------------------------------------------------

Developed by: Anbuselvan.S

GitHub: https://github.com/anbuselvan1519
LinkedIn: https://www.linkedin.com/in/anbu-selvan-1aa819277/

--------------------------------------------------------------------------------
Support
--------------------------------------------------------------------------------

If you find this project useful, please ⭐ the repo and share it with others!
