import ultralytics
from collections import defaultdict
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os


def train(model_path, yaml_path, save_dir):
    model = YOLO(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        device=device,
        workers=0,
        batch=64,
        cache=True,
        amp=False,
        save_dir=save_dir,  # 指定模型保存路径
    )
    time.sleep(10)


def valid(yaml_path, model_path):
    model = YOLO(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_results = model.val(
        data=yaml_path,
        imgsz=640,
        batch=4,
        conf=0.25,
        iou=0.6,
        device=device,
        workers=0,
    )


def display(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO12 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the boxes, track IDs, and class IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks for 'person' class (class_id == 0)
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id == 0:  # Only track 'person' class
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(0, 0, 255),
                        thickness=2,
                    )

            # Display the annotated frame
            cv2.imshow("YOLO12 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def display_webcam(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO12 tracking on the frame, persisting tracks between frames
            # results = model.track(frame, persist=True, classes=[0])
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
            else:
                track_ids = []
                class_ids = []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks for 'person' class (class_id == 0)
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id == 0:  # Only track 'person' class
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(0, 0, 255),
                        thickness=2,
                    )

            # Display the annotated frame
            cv2.imshow("YOLO12 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def test():
    # display("./mytest/1.mp4", "yolo12n.pt")
    display_webcam("yolo12n.pt")


if __name__ == "__main__":
    # todo 将下面的路径修改为正确的路径，如果相对路径出错，需要替换为绝对路径
    # todo 服务器路径/root/autodl-tmp/Yolo
    yaml_path = os.path.join(os.getcwd(), "coco.yaml")
    model_path = os.path.join(os.getcwd(), "yolo12n.pt")
    save_dir = os.path.join(os.getcwd(), "result")
    video_path = os.path.join(os.getcwd(), "mytest/1.mp4")
    result_model_path = os.path.join(save_dir, "runs/detect/train/weights/best.pt")
    train(model_path, yaml_path, save_dir)
    valid(yaml_path, result_model_path)
    display(video_path, result_model_path)
    # test()
