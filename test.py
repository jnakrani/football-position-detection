from ultralytics import YOLO
import cv2

def detect_position(model_path, video_path, output_path):
    """
    Display YOLOv8 detection labels (like QB and C) with custom circle markers on video.

    Args:
        model_path (str): Path to the trained YOLOv8 model.
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video with custom labels.
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result[:6]
            label = results[0].names[int(class_id)]  # Class name

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cv2.circle(frame, (center_x, center_y), 20, (0, 255, 255), -1)
            cv2.putText(frame, label, (center_x - 10, center_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Custom Labeled Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")


model_path = r"model\position_detection.pt"
video_path = r"test_data\151121-Ohio-State-O-vs-Michigan-State_game_2_sideline.mp4"
output_path = "output_custom_labels.mp4"

detect_position(model_path, video_path, output_path)