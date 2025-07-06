import cv2
import numpy as np

# Load model and class labels
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names") as f:
    labels = f.read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# User-defined function for object detection on an image
def detect_objects_in_image(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                cx, cy, bw, bh = int(detection[0]*w), int(detection[1]*h), int(detection[2]*w), int(detection[3]*h)
                x, y = int(cx - bw/2), int(cy - bh/2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)
    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        label = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv4 Image Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_objects_in_image("Canal-Street.jpg")  
