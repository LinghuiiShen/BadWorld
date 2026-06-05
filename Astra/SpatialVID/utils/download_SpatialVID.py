import argparse
from huggingface_hub import hf_hub_download, snapshot_download


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Download SpatialVID dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        choices=["SpatialVID", "SpatialVID-HQ"],
        required=True,
        help="Dataset type to download (SpatialVID or SpatialVID-HQ)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["videos", "annotations", "depths", "metadata", "all"],
        required=True,
        help="Type of data to download (videos, annotations, metadata, all)",
    )
    parser.add_argument(
        "--group_id",
        type=int,
        help="Specific group ID to download (e.g., 'group_1'). If not provided, downloads all groups.",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Local directory to save dataset",
        default="./SpatialVID_data",
    )
    args = parser.parse_args()

    repo_id = f"SpatialVID/{args.repo_id}"

    # Download csv metadata
    if args.type == "metadata":
        hub_path = f"data/train/{args.repo_id.replace('-', '_')}_metadata.csv"
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hub_path,
            local_dir=args.output_dir,
            resume_download=True,
        )
        print(f"Downloaded file '{hub_path}' from {repo_id} to {args.output_dir}")
    # Download specific group
    elif args.group_id:
        hub_path = f"{args.type}/group_{args.group_id:04d}.tar.gz"
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hub_path,
            local_dir=args.output_dir,
            resume_download=True,
        )
        print(f"Downloaded file '{hub_path}' from {repo_id} to {args.output_dir}")
    # Download entire type directory
    elif args.type == "all":
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=args.output_dir,
            resume_download=True,
        )
        print(f"Downloaded entire dataset from {repo_id} to {args.output_dir}")


if __name__ == "__main__":
    main()
