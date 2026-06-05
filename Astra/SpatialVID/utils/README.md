# Utils

- [`convert.py`](convert.py): convert all the paths of videos in a directory to a specific format, like csv.
- [`cut.py`](cut.py): cut videos into clips.
- [`download_SpatialVID.py`](download_SpatialVID.py): download the SpatialVID dataset.
- [`download_YouTube.py`](download_YouTube.py): download videos from YouTube.
- [`evaluate.py`](evaluate.py): evaluate the quality of video reconstructions.
- [`expand_npz.py`](expand_npz.py): get dynamic masks compressed in a npz file.
- [`extract_frames.py`](extract_frames.py): extract frames from videos.
- [`filter.py`](filter.py): filter video clips based on score.
- [`get_clip.py`](get_clip.py): get the clips separated from the video.
- [`get_info.py`](get_info.py): get video information, such as duration and resolution.
- [`get_instructions.py`](get_instructions.py): get motion instructions from camera poses.
- [`get_instructions_enhanced.py`](get_instructions_enhanced.py): an enhanced version to get more detailed and accurate motion instructions from camera poses.
- [`merge_tables.py`](merge_tables.py): merge multiple csv tables into one.
- [`normalize_intrinsics.py`](normalize_intrinsics.py): normalize camera intrinsics.
- [`pack_clip_assets.py`](pack_clip_assets.py): pack all the output files into an npz file for visualization.
- [`quat_to_mat.py`](quat_to_mat.py): convert camera parameters to camera-to-world or world-to-camera matrices.
- [`read_video.py`](read_video.py): read videos using opencv or av.
- [`scene_detect.py`](scene_detect.py): separate videos into clips.
