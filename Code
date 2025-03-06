#2a
import cv2
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for better accuracy

# Open a video source (0 = webcam, or use 'video.mp4')
cap = cv2.VideoCapture(0)

# Get frame dimensions
ret, frame = cap.read()
height, width, _ = frame.shape

# Heatmap storage (accumulated detection points)
heatmap = np.zeros((height, width), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    people_boxes = []
    
    # Extract detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0])  # Class index
            conf = float(box.conf[0])  # Confidence score
            
            # Filter only 'person' class (COCO dataset: class index 0 = person)
            if cls == 0 and conf > 0.5:
                people_boxes.append((x1, y1, x2, y2))
                # Update heatmap at the center of detected person
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                heatmap[center_y, center_x] += 1  # Increment heat at this location
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
    
    # Normalize heatmap for visualization
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay heatmap on the frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

    # Display crowd count
    cv2.putText(overlay, f'People Count: {len(people_boxes)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show output
    cv2.imshow('YOLOv8 Crowd Density + Heatmap', overlay)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
