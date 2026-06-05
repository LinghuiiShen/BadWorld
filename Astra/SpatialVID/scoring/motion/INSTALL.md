# Compiling FFmpeg with NVIDIA GPU Acceleration and VMAF on Ubuntu

This guide provides a comprehensive walkthrough for compiling FFmpeg from source on an Ubuntu system equipped with an NVIDIA GPU. The resulting build will support NVIDIA's hardware encoding/decoding (NVENC/DEC), NPP filters (NVIDIA Performance Primitives), and CUDA-based VMAF (Video Multi-Method Assessment Fusion) for video quality assessment.


## Environment and Versions

Before you begin, ensure your system environment is similar to the configuration below. Version matching is crucial for a successful compilation.
The GPU needs to support HEVC; refer to the [NVIDIA NVDEC Support Matrix](https://en.wikipedia.org/wiki/NVIDIA_Video_Coding_Engine#NVDEC).

-   **GPU**: NVIDIA GeForce RTX 4090 or other compatible models
-   **OS**: Ubuntu 22.04 
-   **NVIDIA Driver Version**: A version compatible with CUDA 12.6
-   **CUDA Version (from `nvidia-smi`)**: `12.x`
-   **CUDA Toolkit Version**: `12.6` (This is the version used for compilation)
-   **Target FFmpeg Version**: `6.1`

**Key Tip**: The version of the `NVIDIA Codec Headers` (`ffnvcodec`) must be compatible with the `CUDA Toolkit` version installed on your system and the version of `FFmpeg` you intend to compile.

## Compilation Steps

Please follow these steps in order.


### Step 1: Install System Dependencies

Update system packages and install required development tools and libraries:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopenjp2-7-dev \
    ninja-build \
    cmake \
    git \
    python3 \
    python3-pip \
    nasm \
    xxd \
    pkg-config \
    curl \
    unzip \
    ca-certificates \
    libnuma-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    vim \
    nvidia-cuda-toolkit
```


### Step 2: Clone Required Repositories

```bash
# Create a working directory (custom path allowed)
mkdir -p ~/ffmpeg-build && cd ~/ffmpeg-build

# Clone nv-codec-headers (NVIDIA codec headers)
git clone https://github.com/FFmpeg/nv-codec-headers.git

# Clone libvmaf (video quality assessment library)
git clone https://github.com/Netflix/vmaf.git
cd vmaf && git checkout master  # Switch to master branch (modify version if needed)
cd ..

# Clone FFmpeg source code
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg && git checkout master  # Switch to master branch (modify version if needed)
cd ..
```


### Step 3: Install nv-codec-headers

```bash
cd nv-codec-headers
make
sudo make install
cd ..
```


### Step 4: Compile and Install libvmaf (with CUDA Support)

1. Install the meson build tool:
   ```bash
   python3 -m pip install meson
   ```

2. Compile and install libvmaf:
   ```bash
   cd vmaf
   meson libvmaf/build libvmaf \
     -Denable_cuda=true \
     -Denable_avx512=true \
     --buildtype release
   ninja -vC libvmaf/build
   sudo ninja -vC libvmaf/build install
   cd ..
   ```

3. Update system library cache:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/x86_64-linux-gnu/
   sudo ldconfig
   ```


### Step 5: Compile and Install FFmpeg (with NVIDIA and libvmaf Support)

```bash
cd FFmpeg

# Configure compilation options (enable CUDA, NVENC, NVDEC, and libvmaf)
./configure \
  --enable-libnpp \
  --enable-nonfree \
  --enable-nvdec \
  --enable-nvenc \
  --enable-cuvid \
  --enable-cuda \
  --enable-cuda-nvcc \
  --enable-libvmaf \
  --enable-ffnvcodec \
  --disable-stripping \
  --extra-cflags="-I/usr/local/cuda/include" \
  --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs/"

# Compile (adjust the number after -j based on CPU cores for faster compilation)
make -j$(nproc)

# Install
sudo make install

cd ..
```


### Step 6: Configure Python Environment

1. Upgrade pip and set up links:
   ```bash
   sudo ln -sf /usr/bin/python3 /usr/bin/python
   python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
   ```

2. Install Python dependencies (assuming project code is cloned locally; replace with actual path):
   ```bash
   # Navigate to the project root directory
   cd /path/to/your/project

   # Install dependencies
   python3 -m pip --no-cache-dir install -r requirements/requirements.txt
   python3 -m pip --no-cache-dir install -r requirements/requirements_scoring.txt || true
   python3 -m pip --no-cache-dir install -r requirements/requirements_annotation.txt || true
   ```


### Step 7: Verify Installation

1. Check FFmpeg version and configuration:
   ```bash
   ffmpeg -version
   ffmpeg -encoders | grep nvenc  # Verify NVENC support
   ffmpeg -decoders | grep nvdec  # Verify NVDEC support
   ffmpeg -filters | grep vmaf    # Verify libvmaf support
   ```

2. If all the above commands output corresponding content correctly, the installation is successful.

## Troubleshooting

### Issue 1: VMAF compilation fails with `vcs_version.h: No such file or directory`

-   **Cause**: This error typically occurs if you downloaded the VMAF source code as a ZIP archive instead of using `git clone`. The build script relies on the `.git` directory to generate version header files.
-   **Solution**: Always use `git clone` to get the source code.
    ```bash
    git clone https://github.com/Netflix/vmaf.git
    ```

### Issue 2: FFmpeg `configure` fails with error about Video Codec SDK version being too low

-   **Error Message**: Something like `ERROR: nvenc requested, but NVIDIA Video Codec SDK 12.1 or later is required.` (The version number may vary).
-   **Cause**: This means the version of `nv-codec-headers` you checked out is not compatible with your NVIDIA driver, CUDA Toolkit, or the version of FFmpeg you are building.
-   **Solution**:
    1.  Carefully re-check your [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) versions.
    2.  Go back to [Step 3: Install NVIDIA Codec Headers](#step-3-install-nvidia-codec-headers) and ensure you `git checkout` the branch that best matches your environment (e.g., `sdk/12.6`).
    3.  Consult the [Official NVIDIA FFmpeg Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/index.html) or the `nv-codec-headers` repository to confirm version compatibility.

## References

-   [VMAF on GitHub](https://github.com/Netflix/vmaf)
-   [FFmpeg Official Source](https://github.com/FFmpeg/FFmpeg/tree/release/6.1)
-   [NVIDIA Codec Headers Source](https://github.com/FFmpeg/nv-codec-headers/tree/sdk/12.6)
-   [Official NVIDIA Guide for Compiling FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/index.html)
