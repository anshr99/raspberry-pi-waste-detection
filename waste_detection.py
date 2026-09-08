from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv5 Nano model
model = YOLO('yolov5n.pt')  # yolov5n.pt is the nano version of YOLOv5, optimized for smaller devices

# Open video capture (for camera, use 0)
cap = cv2.VideoCapture(0)

# Retrieve class names
class_names = model.names

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model(frame)

    # Visualize the results
    for result in results:
        for detection in result.boxes:
            # Get bounding box coordinates, confidence, and class index
            x1, y1, x2, y2 = detection.xyxy[0]
            confidence = float(detection.conf)  # Convert confidence to float
            class_index = int(detection.cls)  # Get the class index as an integer
            label = class_names[class_index]  # Map index to class name
            
            # Classify certain items as waste (you could expand this logic)
            if label in ["bottle", "cup", "plastic"]:
                waste_label = "Waste"
            else:
                waste_label = "Not Waste"

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{waste_label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Waste Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
