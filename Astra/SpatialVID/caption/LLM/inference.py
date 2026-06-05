import os
import time
import queue
from argparse import ArgumentParser
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.api_call import api_call


def get_pose(pose_dir):
    """
    Retrieve and process pose data from extrinsics.npy file
    """
    # Base directory for pose data
    pose_path = os.path.join(pose_dir, 'extrinsics.npy')
    assert os.path.isfile(pose_path), f"Pose file not found: {pose_path}"

    # Load and process the pose file
    poses = np.load(pose_path)

    # Data processing steps
    poses = poses[::5, :, 3]  # Take first row for every 5 rows
    max_value = np.max(poses)
    min_value = np.min(poses)
    min_abs_value = np.min(np.abs(poses))

    # Normalize and convert to integers (minimize integer digits)
    poses = np.round(poses / (max_value - min_value) /
                     min_abs_value).astype(int)

    # Keep only first 3 columns and transpose
    poses = poses[:, :3].T

    # Extract individual axes
    poses1, poses2, poses3 = poses[0], poses[1], poses[2]

    # Convert each axis to string
    poses1_str = ' '.join(map(str, poses1))
    poses2_str = ' '.join(map(str, poses2))
    poses3_str = ' '.join(map(str, poses3))

    # Combine into formatted string
    poses_str = f'x:{poses1_str}\ny:{poses2_str}\nz:{poses3_str}'

    return poses_str


def get_prompt(pose_dir, prompt_dir, vqa_caption, dist_level):
    """
    Construct a prompt by combining content from prompt1.txt, prompt2.txt, VQA caption, and pose data
    """
    # Read prompt components
    p1_file = os.path.join(prompt_dir, 'prompt1.txt')
    p2_file = os.path.join(prompt_dir, 'prompt2.txt')

    with open(p1_file, 'r', encoding='utf-8') as f:
        p1_content = f.read().strip()

    with open(p2_file, 'r', encoding='utf-8') as f:
        p2_content = f.read().strip()

    # Get pose data
    poses = get_pose(pose_dir)

    # Assemble final prompt
    prompt = (f"{p1_content}\nGiven Information:\n{vqa_caption}\n3.Camera Position Data:\n{poses}\n"
              f"\n4.Motion intensity:\n{dist_level}\n{p2_content}")

    return prompt


def process_single_row(args, row):
    """
    Process a single row of data by calling API and saving the result
    """
    # Check if VQA file exists
    vqa_path = os.path.join(args.vqa_path, f"{row['id']}.txt")
    assert os.path.isfile(vqa_path), f"VQA file not found: {vqa_path}"
    # Read VQA caption
    with open(vqa_path, "r") as f:
        vqa_caption = f.read()

    # Skip processing if file already exists
    save_file = os.path.join(args.llm_path, f"{row['id']}.txt")
    if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
        return

    # Call API with retry mechanism
    pose_dir = os.path.join(args.pose_load_dir, row["id"], "reconstructions")
    prompt_text = get_prompt(pose_dir, args.prompt_dir,
                             vqa_caption, row["distLevel"])
    llm_caption = api_call(prompt_text, args.model,
                           args.api_key, args.base_domain)
    assert llm_caption is not None, f"API call failed for id {row['id']}"

    # Save the result with model information
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(llm_caption + f"\n\n6. Qwen model: \n{args.model}")
    return


def worker(args, task_queue, pbar):
    """
    Worker function to process tasks from the queue

    Args:
        task_queue: Queue containing tasks to process
        pbar: Progress bar object for tracking progress
    """
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break

        # Add delay to prevent overwhelming API
        time.sleep(args.wait_time)

        # Process the single row
        process_single_row(args, row)

        # Update progress
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """
    Parse command line arguments

    Returns:
        Parsed arguments object
    """
    parser = ArgumentParser(description='VQA Processing Program')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file')
    parser.add_argument('--pose_load_dir', type=str, required=True,
                        help='Directory to load pose data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--prompt_dir', type=str,
                        default=os.path.join(os.path.dirname(
                            __file__), "vqa_prompt.txt"),
                        help='Path to prompt file')
    parser.add_argument('--model', type=str, default="qwen3-30b-a3b",
                        help='Model name')
    parser.add_argument('--api_key', type=str,
                        default="sk-****",
                        help='API key')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.5,
                        help='Time between requests in seconds')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API base domain')
    return parser.parse_args()


def main():
    """
    Main processing function that handles multiple rows using parallel workers

    Args:
        group_id (str): Identifier for the group
        prompt_dir (str): Directory containing prompt files
        model_file (str): Path to file containing model names
        api_key_file (str): Path to file containing API keys
        num_workers (int): Number of worker threads
        wait_time (float): Time to wait between requests
        base_domain (str): Base domain for API calls
        record_time (bool): Whether to record processing time

    Returns:
        None
    """
    args = parse_args()

    # Validate temporary directory exists
    # Create LLM directory if it doesn't exist
    args.llm_path = os.path.join(args.output_dir, "LLM")
    if not os.path.isdir(args.llm_path):
        os.makedirs(args.llm_path, exist_ok=True)

    # Validate VQA directory exists
    args.vqa_path = os.path.join(args.output_dir, "VQA")
    assert os.path.isdir(
        args.vqa_path), f"VQA directory not found: {args.vqa_path}"

    # Read CSV file containing scene information
    df = pd.read_csv(args.csv_path)

    # Initialize task queue with all rows
    manager = Manager()
    task_queue = manager.Queue()
    for index, row in df.iterrows():
        task_queue.put((index, row))

    # Start processing with progress bar
    with tqdm(total=len(df), desc="LLM Finished") as pbar:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Start worker threads
            futures = [executor.submit(worker, args, task_queue, pbar)
                       for _ in range(args.num_workers)]
            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
