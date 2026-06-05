import os
import time
import json
import queue
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
import concurrent.futures
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.api_call import api_call


def parse_category_tags(tag_caption):
    """
    Parse API response to structured category data using camelCase naming convention
    """
    lines = [line.strip()
             for line in tag_caption.strip().split('\n') if line.strip()]

    # Initialize category data with default values
    category_data = {
        "sceneType": {
            "first": "Unknown",
            "second": "Unknown"
        },
        "lighting": "Unknown",
        "timeOfDay": "Unknown",
        "weather": "Unknown",
        "crowdDensity": "Unknown"
    }

    # Parse each line to extract category information
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("primary scene type:"):
            category_data["sceneType"]["first"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("secondary scene type:"):
            category_data["sceneType"]["second"] = line.split(":", 1)[
                1].strip()
        elif line_lower.startswith("lighting:"):
            category_data["lighting"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("time of day:"):
            category_data["timeOfDay"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("weather:"):
            category_data["weather"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("crowd density:"):
            category_data["crowdDensity"] = line.split(":", 1)[1].strip()

    return category_data


def process_single_row(args, json_file):
    """
    Process a single JSON file to add category tags via API call
    """
    # Check if CategoryTag field already exists
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Skip if CategoryTag already exists
    if "CategoryTag" in data:
        return
    description = data['SceneDesc']

    prompt_text = args.prompt_text + description

    # Call API to get category tags with retry mechanism
    tag_caption = api_call(prompt_text, args.model,
                           args.api_key, args.base_domain)
    assert tag_caption is not None, f"API call failed for file {json_file}"

    # Parse and add category tags to the JSON file
    category_tag = parse_category_tags(tag_caption)

    # Merge new data with existing data
    data["CategoryTag"] = category_tag

    # Overwrite file with updated data
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def worker(args, task_queue, pbar):
    while True:
        try:
            index, json_file = task_queue.get(timeout=1)
        except queue.Empty:
            break

        time.sleep(args.wait_time)
        process_single_row(args, json_file)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Category Tag Processing Program')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the CSV file')
    parser.add_argument('--json_load_dir', type=str, required=True,
                        help='Directory containing JSON files')
    parser.add_argument('--prompt_file', type=str,
                        default="prompt.txt",
                        help='Path to prompt file')
    parser.add_argument('--model', type=str, default="qwen3-30b-a3b",
                        help='Model name')
    parser.add_argument('--api_key', type=str,
                        default="sk-****",
                        help='API key')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='Time interval between requests per thread (seconds)')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API base domain')
    return parser.parse_args()


def main():
    """
    Process a group of JSON files using multiple threads to add category tags
    """
    args = parse_args()

    df = pd.read_csv(args.csv_path)

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        args.prompt_text = f.read().strip()

    # Initialize task queue and add all files to process
    manager = Manager()
    task_queue = manager.Queue()
    for index, row in df.iterrows():
        clip_id = row['id']
        json_file = os.path.join(args.json_load_dir, f"{clip_id}.json")
        task_queue.put((index, json_file))

    # Start processing with progress bar
    with tqdm(total=task_queue.qsize(), desc="Tags Completed") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Start worker threads
            futures = [executor.submit(worker, args, task_queue, pbar)
                       for _ in range(args.num_workers)]

            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
