import os
import concurrent.futures
from multiprocessing import Manager
import queue
import pandas as pd
from tqdm import tqdm
import argparse
import time
import base64
import cv2
from glob import glob
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.api_call import api_call


def encode_image(image_path):
    """
    Resizes an image to 640x360 and encodes it as a Base64 string with data URI prefix.
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Resize image to standard dimensions (640x360)
    resized_image = cv2.resize(image, (640, 360))

    # Encode image as JPEG and convert to Base64
    _, buffer = cv2.imencode('.jpeg', resized_image)
    base64_data = base64.b64encode(buffer).decode("utf-8")

    # Return with data URI format for API compatibility
    return f"data:image/jpeg;base64,{base64_data}"


def get_prompt(fig_dir, prompt_text):
    """
    Load key frames from a video, constructs a multimodal request, and calls the API.
    """
    # Get frames from directory
    frames = sorted(glob(f"{fig_dir}/*.jpg"))[::5]

    # Construct multimodal input content
    messages_content = []

    # Add encoded images to request content
    for frame in frames:
        try:
            encoded_frame = encode_image(frame)
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": encoded_frame}
            })
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return None

    # Add text prompt to request content
    messages_content.append({"type": "text", "text": prompt_text})

    return messages_content


def process_single_row(args, row):
    """
    Process a single row: call the VQA API and save the result for one scene.
    Handles retries and error logging.
    """
    save_path = os.path.join(args.output_dir, "VQA")
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{row['id']}.txt")
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
        # Skip if already exists
        return
    # Call API
    fig_dir = os.path.join(args.fig_load_dir, row['id'], "img")
    prompt_text = get_prompt(fig_dir, args.prompt_text)
    vqa_caption = api_call(prompt_text, args.model,
                           args.api_key, args.base_domain)
    assert vqa_caption is not None, f"API call failed for id {row['id']}"
    # Save result
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(vqa_caption)


def worker(args, task_queue, pbar):
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        time.sleep(args.wait_time)
        process_single_row(args, row)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """
    Parse command line arguments for VQA batch processing.
    """
    parser = argparse.ArgumentParser(description='VQA batch processing script')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV file path')
    parser.add_argument('--fig_load_dir', type=str, required=True,
                        help='Directory to load figures')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--prompt_file', type=str,
                        default="vqa_prompt.txt",
                        help='Prompt file path')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash",
                        help='Model name')
    parser.add_argument('--api_key', type=str,
                        default="sk-****",
                        help='API key')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='Request interval for each thread (seconds)')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API base domain')
    return parser.parse_args()


def main():
    """
    Batch process all scenes in a group: call VQA API for each row in the CSV.
    Uses a thread pool for concurrency and supports timing.
    """
    args = parse_args()

    df = pd.read_csv(args.csv_path)

    # Read prompt text
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        args.prompt_text = f.read().strip()

    manager = Manager()
    task_queue = manager.Queue()
    for index, row in df.iterrows():
        task_queue.put((index, row))

    with tqdm(total=len(df), desc=f"VQA Finished") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for worker_id in range(args.num_workers):
                futures.append(executor.submit(worker, args, task_queue, pbar))
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
