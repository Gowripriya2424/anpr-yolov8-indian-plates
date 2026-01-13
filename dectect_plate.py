from ultralytics import YOLO
import cv2
import os

def detect_number_plate(image_path, output_path):
    """
    Detects number plates in an image using YOLOv8 and saves the output image.
    """

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Input image not found.")

    # Perform detection
    results = model(image)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                "Number Plate",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

    print("Number plate detection completed successfully.")


if __name__ == "__main__":
    detect_number_plate(
        image_path="input.jpg",
        output_path="sample_outputs/detected_plate.jpg"
    )
