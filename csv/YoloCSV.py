import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.4
MAX_OBJECTS = 10

model = YOLO('yolov8n.pt')

def process_vid(video_path, vid, frame_no):
    tracker = DeepSort(max_age=8)
    rows = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video:", video_path)
        return rows

    frame_count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Event is only at frame_no
        if frame_count == frame_no:
            target_val = 1
        else:
            target_val = 0

        resized_frame = cv2.resize(frame, (640, 360))
        yolo_dets = model(resized_frame, verbose=False)[0]

        results = []
        frame_height, frame_width = frame.shape[:2]

        # Collect detections
        for data in yolo_dets.boxes.data.tolist():
            confidence = data[4]
            class_id = int(data[5])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = [
                int(coord * (frame_width / 640 if i % 2 == 0 else frame_height / 360))
                for i, coord in enumerate(data[:4])
            ]
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Update tracker once per frame
        tracks = tracker.update_tracks(results, frame=frame)

        detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            detections.append(f"[{xmin},{ymin},{xmax},{ymax}]")

        # Pad or trim detections to MAX_OBJECTS
        if len(detections) < MAX_OBJECTS:
            detections.extend(["0"] * (MAX_OBJECTS - len(detections)))
        else:
            detections = detections[:MAX_OBJECTS]

        rows.append([vid, frame_count] + detections + [target_val])

    cap.release()
    return rows


def generate_csv(video_dir, df, output_csv):
    all_rows = []

    for idx, row in df.iterrows():
        vid = row['Video_Id']
        frame_no = row['Frame_No']
        if np.isnan(frame_no):
            print(f"[{idx+1}/{len(df)}] Not Processing video {vid} ...")
            print(f"skipping {vid} no frame_no")
            continue


        video_path = f"{video_dir}/{vid}"

        print(f"[{idx+1}/{len(df)}] Processing video {vid} ...")

        rows = process_vid(video_path, vid, frame_no)
        all_rows.extend(rows)

        print(f"✅ Finished {vid}, frames processed: {len(rows)}")

    columns = ["video_id", "frame"] + [f"object_{i}" for i in range(MAX_OBJECTS)] + ["target"]
    pd.DataFrame(all_rows, columns=columns).to_csv(output_csv, index=False)
    print(f"\n📂 CSV saved to: {output_csv}")


# Usage
def main():
    df = pd.read_csv('/content/testing - Sheet1.csv')
    video_dir = '/content/drive/MyDrive/NiAD_LargeVideos/Testing'
    output_csv = '/content/output.csv'
    generate_csv(video_dir, df, output_csv)

if __name__ == "__main__":
    main()

