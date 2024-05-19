from collections import defaultdict

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

video_path = r'C:\Users\NGUYEN VAN MANH\Downloads\atgt.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
model = YOLO('best2.pt')

# Set buffer size to a smaller value (e.g., 1) to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

road_zoneA = np.array([[0, 503], [302, 565], [457, 94], [357, 86], [5, 503]], np.int32)
road_zoneB = np.array([[327, 556], [701, 563], [575, 81], [471, 81], [328, 556]], np.int32)
road_zoneC = np.array([[715, 540], [1019, 493], [683, 75], [587, 84], [719, 540]], np.int32)

zoneAcounter = []
zoneBcounter = []
zoneCcounter = []

# Store the track history
track_history = defaultdict(lambda: [])

# Open a video sink for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "output_single_line.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (1920, 1080))
    cv2.line(frame,(0, 503),(302, 565),(0,0,255),thickness=1)
    cv2.line(frame,(457, 94),(357, 86),(0,0,255),thickness=1)
    cv2.line(frame,(327, 556),(701, 563),(0,255,255),thickness=1)
    cv2.line(frame,(575, 81),(471, 81),(0,255,255),thickness=1)
    cv2.line(frame,(715, 540),(1019, 493),(255,0,0),thickness=1)
    cv2.line(frame,(683, 75),(587, 84),(255,0,0),thickness=1)


    # Hiển thị số lượng phương tiện giao thông trên từng lane
    cv2.circle(frame, (870, 90), 15, (0, 0, 255), -1)
    cv2.circle(frame, (870, 130), 15, (0, 255, 255), -1)
    cv2.circle(frame, (870, 170), 15, (255, 0, 0), -1)
    cvzone.putTextRect(frame, f'LANE A Vehicles ={len(zoneAcounter)}', [900, 99], thickness=4, scale=2, border=2)
    cvzone.putTextRect(frame, f'LANE B Vehicles ={len(zoneBcounter)}', [900, 140], thickness=4, scale=2, border=2)
    cvzone.putTextRect(frame, f'LANE C Vehicles ={len(zoneCcounter)}', [900, 180], thickness=4, scale=2, border=2)

    # Run YOLOv8 tracking on the frame
    # results = model.track(frame, classes=[0, 1, 2, 3], persist=True, save=True, tracker="bytetrack.yaml")
    results = model.track(frame, classes=[2, 3, 5, 7], persist=True, tracker="bytetrack.yaml")

    boxes = results[0].boxes.xywh.cpu()
    track_ids = []
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    current_detections = np.empty([0, 5])

    # Plot the tracks and count objects in each lane
    for box, track_id in zip(boxes, track_ids):
        x_center, y_center, width, height = box[0], box[1], box[2], box[3]
        cx, cy = x_center, y_center  # Tính tâm của bounding box
        track = track_history[track_id]
        track.append((float(cx), float(cy)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)
        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Kiểm tra xem bounding box thuộc lane nào và cập nhật đếm
        point = tuple(map(int, [cx, cy]))
        # Kiểm tra lane A
        if cv2.pointPolygonTest(road_zoneA, point, False) > 0:
            if track_id not in zoneAcounter:
                zoneAcounter.append(track_id)
        elif track_id in zoneAcounter:
            zoneAcounter.remove(track_id)

        # Kiểm tra lane B
        if cv2.pointPolygonTest(road_zoneB, point, False) > 0:
            if track_id not in zoneBcounter:
                zoneBcounter.append(track_id)
        elif track_id in zoneBcounter:
            zoneBcounter.remove(track_id)

        # Kiểm tra lane C
        if cv2.pointPolygonTest(road_zoneC, point, False) > 0:
            if track_id not in zoneCcounter:
                zoneCcounter.append(track_id)
        elif track_id in zoneCcounter:
            zoneCcounter.remove(track_id)


    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
