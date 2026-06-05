import zipfile
import numpy as np
import OpenEXR


def read_depth(zip_file_path):
    """
    Read depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    depth_data_list = []
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    # Sometimes EXR loader might fail, we return all nan maps.
                    assert valid_width > 0 and valid_height > 0
                    depth_data_list.append(
                        np.full((valid_height, valid_width), np.nan, dtype=np.float32))
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(
                    channels[0], dtype=np.float16).reshape((height, width))
                depth_data_list.append(depth_data.astype(np.float32))
    # Note that the depth with a negative value is an invalid depth.
    # It can be set to the farthest point or other operations.
    depth_array = np.array(depth_data_list)
    depth_array_safe = np.where(depth_array == 0, 1e-12, depth_array)
    return 1.0 / depth_array_safe
