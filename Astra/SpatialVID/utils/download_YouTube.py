"""
Utility script to download YouTube videos using yt-dlp with support for concurrency and sharding.
Adapted from https://huggingface.co/Ligeng-Zhu/panda70m-download

running script: python download_YouTube.py --csv="$csv_file" # this csv file must contains 'url' column

if you want to download a specific youtube video, consider using:
- yt-dlp -F --list-formats https://www.youtube.com/watch\?v\=omP01s7RUSA # --proxy 127.0.0.1:xxxx --cookies cookies.txt

run 'ls -l /path/to/folder/*.json | wc -l' for counting the videos already downloaded

Customization Guide: 
    For customizing download settings (such as video format, cookie configurations like automatic Chrome cookie retrieval or custom cookie file usage), 
    refer to the official documentation at https://github.com/yt-dlp/yt-dlp-wiki.
"""

import sys, os, os.path as osp
import yt_dlp
import asyncio
from concurrent.futures import ProcessPoolExecutor
import fire
import pandas as pd
import json
import time


def ytb_download(url, json_info, output_dir="ytb_videos/"):
    """
    Download a specified YouTube video using yt-dlp and save related metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    uid = url.split("?v=")[-1]
    yt_opts = {
        "format": "bv[height=720][ext=mp4]"
        # "format": "bv[height=720]",  # Download the best quality available
        # "format": "bv[height=720][ext=mp4][vcodec!^=av]",
        # "proxy": "127.0.0.1:xxxx",
        "outtmpl": osp.join(output_dir, f"{uid}.%(ext)s"),  # Output template
        # "cookiesfrombrowser": "chrome",  # Use Chrome's cookies automatically
        # "cookiefile": "cookies.txt",  # Use a custom cookies file
        # "postprocessors": [
        #     {
        #         "key": "FFmpegVideoConvertor",
        #         "preferedformat": "mp4",  # Convert video to mp4 format (slow)
        #     }
        # ],
        # "verbose" : True,
        "abort-on-error": True,  # Abort downloading when an error occurs
        "retries": 60,  # Number of retries
        "ffmpeg_location": "/usr/bin/ffmpeg",  # Path to ffmpeg
        "quiet": True,  # Suppress output
        "sleep-requested": 5,  # Sleep for 1.25 seconds between requests
        "min-sleep-interval": 60,
        "max-sleep-interval": 90,
    }
    video_path_mp4 = osp.join(output_dir, f"{uid}.mp4")
    video_path_webm = osp.join(output_dir, f"{uid}.webm")
    meta_path = osp.join(output_dir, f"{uid}.json")
    if (osp.exists(video_path_mp4) or osp.exists(video_path_webm)) and osp.exists(
        meta_path
    ):
        print(f"\033[91m{uid} already labeled.\033[0m")
        return 0
    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([url])
        with open(meta_path, "w") as fp:
            json.dump(json_info, fp, indent=2)
        return 0
    # exception logs
    except Exception as e:
        print(f"\033[91mError downloading {url}: {e}\033[0m")
        err_map = {
            "Requested format is not available": "z0322_dld_format_noavailable.log",
            "removed by": "z0322_dld_removed_by.log",
            "Private video": "z0322_dld_private_video.log",
        }
        for key, log_file in err_map.items():
            if key in str(e):
                with open(osp.join(output_dir, f"{log_file}"), "a") as f:
                    f.write(f"{url}\n")
                break
        else:
            with open(osp.join(output_dir, f"z0322_dld_othererr.log"), "a") as f:
                f.write(f"{url}, {str(e)}\n")
        return -1


async def main(csv_path, output_dir, max_workers=10, shards=0, total=-1, limit=False):
    """
    Batch download YouTube videos specified in a CSV file, supporting sharding and concurrency.
    """
    PPE = ProcessPoolExecutor(max_workers=max_workers)
    loop = asyncio.get_event_loop()
    df = pd.read_csv(csv_path)
    csv_path = os.path.basename(csv_path)
    output_dir = f'{output_dir}/{csv_path.split(".")[0]}'
    data_list = list(df.iterrows())
    if total > 0:
        chunk = len(data_list) // total
        begin_idx = shards * chunk
        end_idx = (shards + 1) * chunk if shards < total - 1 else len(data_list)
        data_list = data_list[begin_idx:end_idx]
    print(f"download total {len(data_list)} videos")
    tasks = []
    for idx, (index, row) in enumerate(data_list):
        video_url = row["url"]
        # json_info = {"caption": row["caption"]}
        json_info = {"caption": ''} # for file checking.
        tasks.append(
            loop.run_in_executor(PPE, ytb_download, video_url, json_info, output_dir)
        )
        if limit and idx >= 20:
            break
    res = await asyncio.gather(*tasks)
    print(f"[{sum(res)} / {len(res)}]")


def entry(
    csv="meta_data_sample_500.csv",
    output_dir="path/to/output",
    shards=0,
    total=-1,
    limit=False,
    max_workers=2,
):
    """
    Command line entry function, supports fire invocation.
    """
    print(csv, output_dir, shards, total, max_workers)
    start_time = time.time()
    print(
        f"\033[92mStarting execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\033[0m"
    )
    asyncio.run(
        main(
            csv,
            output_dir,
            max_workers=max_workers,
            shards=shards,
            total=total,
            limit=limit,
        )
    )
    end_time = time.time()
    print(
        f"\033[92mFinished execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\033[0m"
    )
    print(f"\033[92mTotal execution time: {end_time - start_time:.2f} seconds\033[0m")


def add_download(csv_path):
    """
    Download missing videos according to the new_vid_path field in the CSV file.
    """
    data = pd.read_csv(csv_path)
    unique_ids = data['YouTube id'].unique()

    for uid in unique_ids:
        video_url = f"https://www.youtube.com/watch?v={uid}"
        ytb_download(video_url, json_info={}, output_dir="videos/")
        print(f"Downloaded {video_url}")


if __name__ == "__main__":
    # Call entry function via command line arguments
    
    fire.Fire(entry)
    # for supplement download: add_download(csv_path='xxx.csv')
