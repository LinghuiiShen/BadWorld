import os
import json
import re
import queue
import argparse
import pandas as pd
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def parse_text_to_json(text):
    """
    Parses text in a specific format into a JSON structure.
    """
    # Define mapping between text labels and JSON keys
    labels = {
        "Camera Motion Caption": "OptCamMotion",
        "Scene Abstract Caption": "SceneSummary",
        "Main Motion Trend Summary": "MotionTrends",
        "Scene Keywords": "SceneTags",
        "Immersive Shot Summary": "ShotImmersion",
        "Qwen model": "LLM"
    }

    # Initialize result dictionary with empty values
    result = {key: "" for key in labels.values()}
    current_label = None
    current_content = []

    # Process text line by line
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if current line contains any label
        for label, json_key in labels.items():
            if label in line:
                # Find position of first letter after the label
                start_pos = line.find(label) + len(label)
                # Skip non-alphabet characters
                while start_pos < len(line) and not line[start_pos].isalpha():
                    start_pos += 1

                content = line[start_pos:].strip()

                # Process content if it exists after the label
                if content:
                    if json_key in ["MotionTrends", "SceneTags"]:
                        # Split by commas (both Chinese and English), preserve spaces in phrases
                        items = re.split(r'[，,]\s*', content)
                        result[json_key] = [item.strip()
                                            for item in items if item.strip()]
                    else:
                        result[json_key] = content

                    current_label = None
                    current_content = []
                else:
                    # No content after label, continue reading subsequent lines
                    current_label = json_key
                    current_content = []

                break
        else:
            # If collecting content for a label
            if current_label:
                # Check if line is not empty
                if line:
                    # Find first letter in line
                    start_pos = 0
                    while start_pos < len(line) and not line[start_pos].isalpha():
                        start_pos += 1

                    if start_pos < len(line):
                        current_content.append(line[start_pos:])
                else:
                    # Empty line indicates end of current label content
                    content = ' '.join(current_content).strip()

                    if current_label in ["MotionTrends", "SceneTags"]:
                        # Split by commas (both Chinese and English), preserve spaces in phrases
                        items = re.split(r'[，,]\s*', content)
                        result[current_label] = [item.strip()
                                                 for item in items if item.strip()]
                    else:
                        result[current_label] = content

                    current_label = None
                    current_content = []

        i += 1

    # Handle Qwen model label which might extend to the end
    if current_label == "LLM" and current_content:
        content = ' '.join(current_content).strip()
        result[current_label] = content

    return result


def vqa_parse_text_to_json(text):
    """
    Parses text containing Camera Motion Caption and Scene Description into JSON format.
    """
    result = {
        "CamMotion": "",
        "SceneDesc": ""
    }

    # Process Camera Motion Caption - from first letter after label to newline
    camera_pattern = r'Camera Motion Caption:\s*(\w[\s\S]*?)(?=\n|$)'
    camera_match = re.search(camera_pattern, text)
    if camera_match:
        result["CamMotion"] = camera_match.group(1).strip()

    # Process Scene Description - from first letter after label to end of text
    scene_pattern = r'Scene Description:\s*(\w[\s\S]*)$'
    scene_match = re.search(scene_pattern, text, re.DOTALL)
    if scene_match:
        result["SceneDesc"] = scene_match.group(1).strip()

    return result


def process_single_row(args, clip_id):
    """
    Processes VQA and LLM captions for a single clip and merges them into one JSON file.
    """
    # Define file paths
    vqa_path = os.path.join(args.load_dir, "VQA", f"{clip_id}.txt")
    assert os.path.exists(vqa_path), f"VQA path does not exist: {vqa_path}"
    llm_path = os.path.join(args.load_dir, "LLM", f"{clip_id}.txt")
    assert os.path.exists(llm_path), f"LLM path does not exist: {llm_path}"
    output_path = os.path.join(args.output_dir, f"{clip_id}.json")

    # Skip if output file already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return

    # Read VQA file content
    with open(vqa_path, 'r', encoding='utf-8') as f:
        vqa_text = f.read()

    # Read LLM file content
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_text = f.read()

    # Parse text content to JSON
    vqa_json = vqa_parse_text_to_json(vqa_text)
    llm_json = parse_text_to_json(llm_text)

    # Merge JSON objects
    combined_json = {**vqa_json, **llm_json}

    # Save merged JSON to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_json, f, ensure_ascii=False, indent=2)


def worker(args, task_queue, pbar):
    while True:
        try:
            idx, clip_id = task_queue.get(timeout=1)
        except queue.Empty:
            break

        process_single_row(args, clip_id)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge VQA and LLM caption data')
    parser.add_argument('--csv_path', type=str,
                        required=True, help='Path to the CSV file')
    parser.add_argument('--load_dir', type=str, required=True,
                        help='Directory containing caption files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save merged JSON files')
    parser.add_argument('--num_workers', type=int,
                        default=32, help='Number of worker threads')
    return parser.parse_args()


def main():
    """
    Processes all scenes in the specified batch.
    """
    args = parse_args()

    df = pd.read_csv(args.csv_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # Use multiprocessing manager for thread-safe queue
    manager = Manager()
    task_queue = manager.Queue()

    # Add tasks to queue
    for index, row in df.iterrows():
        task_queue.put((index, row['id']))

    # Start multi-threaded processing with progress bar
    with tqdm(total=len(df), desc="Processing progress") as pbar:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for _ in range(args.num_workers):
                futures.append(executor.submit(worker, args, task_queue, pbar))

            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
