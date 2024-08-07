import cv2
import numpy as np
import time
import sys
import os

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
file = "../data/man-walking.mp4"


def create_tracker():
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    ]
    tracker_type = tracker_types[7]

    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker


tracker = create_tracker()
video = cv2.VideoCapture(file)
if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)


while True:
    ret, frame = video.read()
    if not ret:
        break

    start_time = time.time()
    object_found, bbox = tracker.update(frame)
    end_time = time.time()

    print("tracked ", bbox, " in ", (end_time - start_time) * 1000, "ms")
    if object_found:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
