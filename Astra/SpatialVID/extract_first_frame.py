import os
import cv2
from glob import glob

video_dir = "SpatialVID_HQ/videos/SpatialVID/videos/group_0001" 
out_dir = "SpatialVID/images/group_0001"

os.makedirs(out_dir, exist_ok=True)

videos = glob(os.path.join(video_dir, "*.mp4"))

for v in videos:
    name = os.path.splitext(os.path.basename(v))[0]
    out_path = os.path.join(out_dir, f"{name}.jpg")

    if os.path.exists(out_path):
        print("skip", name)
        continue

    cap = cv2.VideoCapture(v)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(out_path, frame)
        print("ok", name)
    else:
        print("failed", name)

print("done")