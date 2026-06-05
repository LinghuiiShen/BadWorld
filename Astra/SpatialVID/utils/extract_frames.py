"""
Video frame extraction utility with parallel processing support.
"""

import os
import sys
import cv2
import av
import argparse
import pandas as pd
import queue
import concurrent.futures
from multiprocessing import Manager
from tqdm import tqdm


def extract_frames_opencv(
    video_path, output_dir, interval, frame_start, num_frames, target_size=None
):
    """Extract frames from video at specified intervals"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    # Extract frames
    for frame in range(num_frames):
        ret, image = cap.read()
        if not ret:
            break

        # Save frame at specified intervals
        if frame % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame:06d}.jpg")
            if target_size is not None:
                h, w = image.shape[:2]
                # Adaptively adjust target size based on video orientation
                # For portrait videos (height > width), swap width and height of target size
                if h > w:  # Portrait video
                    target_w, target_h = target_size[1], target_size[0]
                else:  # Landscape video
                    target_w, target_h = target_size[0], target_size[1]
                image = cv2.resize(image, (target_w, target_h))
            cv2.imwrite(frame_filename, image)

    cap.release()


def extract_frames_av(
    video_path, output_dir, interval, frame_start, num_frames, target_size=None
):
    """
    Extract frames from video at specified intervals using PyAV backend.
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Open video file
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
    except Exception as e:
        print(f"Error: Could not open video file {video_path}. Reason: {e}")
        return

    # Get video properties
    fps = float(stream.average_rate)
    time_base = stream.time_base
   
    target_sec = frame_start / fps
    # Set a small tolerance (e.g., half a frame time) to prevent frame loss due to floating-point precision issues
    epsilon = 0.5 / fps 

    # Seek to the target start time
    if frame_start > 0:
        target_pts = int(target_sec / time_base)
        container.seek(target_pts, stream=stream, backward=True)

    count = 0
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
            
        current_sec = frame.pts * time_base
        if current_sec < (target_sec - epsilon):
            continue

        if count >= num_frames:
            break
        
        # Save frame at specified intervals
        if count % interval == 0:
            image = frame.to_ndarray(format='bgr24')
            frame_filename = os.path.join(output_dir, f"frame_{count:06d}.jpg")
            if target_size is not None:
                if isinstance(target_size, str):
                     w, h = map(int, target_size.split('*'))
                     target_size = (w, h)
                # Adaptively adjust target size based on video orientation
                h, w = image.shape[:2]
                # For portrait videos (height > width), swap width and height of target size
                if h > w:  # Portrait video
                    target_w, target_h = target_size[1], target_size[0]
                else:  # Landscape video
                    target_w, target_h = target_size[0], target_size[1]
                image = cv2.resize(image, (target_w, target_h))
            cv2.imwrite(frame_filename, image)

        count += 1

    container.close()


def process_single_row(row, row_index, args):
    """Process a single video row to extract frames"""
    video_path = row["video_path"]
    frame_start = row.get("frame_start", 0)
    num_frames = row["num_frames"]
    output_dir = os.path.join(args.output_dir, row["id"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, "img")

    # Calculate frame extraction interval
    if args.interval is None:
        interval = row["num_frames"] // 3  # Extract 3 frames by default
    elif args.interval == 0:
        interval = 1  # Extract every frame
    else:
        interval = int(args.interval * row["fps"])

    if args.backend == "opencv":
        extract_frames_opencv(
            video_path, output_dir, interval, frame_start, num_frames, args.target_size
        )
    elif args.backend == "av":
        extract_frames_av(
            video_path, output_dir, interval, frame_start, num_frames, args.target_size
        )


def worker(task_queue, progress_queue, args):
    """Worker function for parallel frame extraction"""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        process_single_row(row, index, args)
        progress_queue.put(index)
        task_queue.task_done()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Extract frames from video files")
    parser.add_argument(
        "--csv_path", type=str, help="Path to CSV file with video csvdata"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extract_frames",
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Frame extraction interval in seconds (set to 0 to extract every frame)",
    )
    parser.add_argument(
        "--target_size",
        type=str,
        default=None,
        help="Resize frames to size (width*height). For portrait videos (h>w), dimensions will be automatically swapped to (height*width) to maintain correct orientation.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="opencv",
        choices=["opencv", "av"],
        help="Backend for video reading",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    return parser.parse_args()


def main():
    """Main function to process frame extraction"""
    args = parse_args()

    # Parse target size if provided
    if args.target_size is not None:
        args.target_size = tuple(map(int, args.target_size.split("*")))

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load video csvdata
    csv = pd.read_csv(args.csv_path)

    if args.disable_parallel:
        # Sequential processing
        for index, row in tqdm(
            csv.iterrows(), total=len(csv), desc="Processing videos"
        ):
            process_single_row(row, index, args)
    else:
        # Parallel processing
        num_workers = args.num_workers if args.num_workers else os.cpu_count() or 1

        manager = Manager()
        task_queue = manager.Queue()
        progress_queue = manager.Queue()

        # Add tasks to queue
        for index, row in csv.iterrows():
            task_queue.put((index, row))

        # Execute workers
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = []
            for _ in range(num_workers):
                future = executor.submit(worker, task_queue, progress_queue, args)
                futures.append(future)

            processed = 0
            total_tasks = len(csv)
            with tqdm(total=total_tasks, desc="Processing videos") as pbar:
                while processed < total_tasks:
                    try:
                        progress_queue.get(timeout=1)
                        processed += 1
                        pbar.update(1)
                    except queue.Empty:
                        if all(f.done() for f in futures) and progress_queue.empty():
                            break

            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
