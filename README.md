# 💦 OpenSplat

A free and open source implementation of 3D [gaussian splatting](https://www.youtube.com/watch?v=HVv_IQKlafQ) written in C++, focused on being portable, lean and fast.

<img src="https://github.com/pierotofy/OpenSplat/assets/1951843/c9327c7c-31ad-402d-a5a5-04f7602ca5f5" width="49%" />
<img src="https://github.com/pierotofy/OpenSplat/assets/1951843/eba4ae75-2c88-4c9e-a66b-608b574d085f" width="49%" />

OpenSplat takes camera poses + sparse points in [COLMAP](https://colmap.github.io/), [OpenSfM](https://github.com/mapillary/OpenSfM), [ODX](https://github.com/WebODM/ODX),  [OpenMVG](https://github.com/OpenMVG/OpenMVG) or [nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html) project format and computes a [scene file](https://drive.google.com/file/d/12lmvVWpFlFPL6nxl2e2d-4u4a31RCSKT/view?usp=sharing) (.ply, .splat, .spz, or .rad) that can be later imported for [viewing](https://antimatter15.com/splat/?url=https://splat.uav4geo.com/banana.splat), editing and rendering in other [software](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#open-source-implementations).

Graphics card recommended, but not required! OpenSplat runs the fastest on NVIDIA, AMD and Apple (Metal) GPUs, but can also run entirely on the CPU (~100x slower).

Commercial use allowed and encouraged under the terms of the [AGPLv3](https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0). ✅

We even have a [song](https://youtu.be/1bma7XJkoDM) 🎵

## Getting Started

If you're on Windows, you can [buy](http://sites.fastspring.com/masseranolabs/product/opensplatforwindows) the pre-built program. This saves you time and helps support the project ❤️. Then jump directly to the [run](#run) section. As an alternative, check the [build](#build) section below.

If you're on macOS or Linux check the [build](#build) section below. 

## Build

You can build OpenSplat with or without GPU support.

Requirements for all builds:

 * **OpenCV**: `sudo apt install libopencv-dev` should do it.
 * **libtorch**: See instructions below.

### CPU

For libtorch visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". For compute platform you can select "CPU".

 Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
 ```

### CUDA

Additional requirement:

 * **CUDA**: Make sure you have the CUDA compiler (`nvcc`) in your PATH and that `nvidia-smi` is working. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
 
 For libtorch visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". Make sure to match your version of CUDA if you want to leverage GPU support in libtorch.
 
 Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
 ```

### ROCm via HIP

Additional requirement:

* **ROCm**: Make sure you have the ROCm installed at /opt/rocm. https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html

For libtorch visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". Make sure to match your version of ROCm (5.7) if you want to leverage AMD GPU support in libtorch.

Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 export PYTORCH_ROCM_ARCH=gfx906
 cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
 make
 ```

In addition, you can leverage Jinja to build the project

```bash
cmake -GNinja -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
jinja
```

### Windows

There's several ways to build on Windows, but this particular configuration has been confirmed to work:

* Visual Studio 2022 C++
* https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-windows-x86_64.msi
* https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe
* https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip
* https://github.com/opencv/opencv/releases/download/4.9.0/opencv-4.9.0-windows.exe

Then run:

```console
"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
git clone https://github.com/pierotofy/OpenSplat OpenSplat
cd OpenSplat
md build
cd build
cmake -DCMAKE_PREFIX_PATH=C:/path_to/libtorch_2.1.2_cu11.8/libtorch -DOPENCV_DIR=C:/path_to/OpenCV_4.9.0/build -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Optional: Edit cuda target (only if required) before `cmake --build .`

C:/path_to/OpenSplat/build/gsplat.vcxproj
for example: arch=compute_75,code=sm_75

### macOS

If you're using [Homebrew](https://brew.sh), you can install Cmake/OpenCV/Pytorch by running:

```bash
brew install cmake
brew install opencv
brew install pytorch
```

You will also need to install Xcode and the Xcode command line tools to compile with metal support (otherwise, OpenSplat will build with CPU acceleration only):
1. Install Xcode from the Apple App Store.
2. Install the command line tools with `xcode-select --install`. This might do nothing on your machine.
3. If `xcode-select --print-path` prints `/Library/Developer/CommandLineTools`,then run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`.
4. On recent Xcode versions the Metal toolchain is a separate download. If `xcrun -sdk macosx metal --version` fails, run `xcodebuild -downloadComponent MetalToolchain`.

Then run:

```
git clone https://github.com/pierotofy/OpenSplat OpenSplat
cd OpenSplat
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(sysctl -n hw.logicalcpu)
./opensplat
```

On macOS `GPU_RUNTIME` defaults to `MPS` (metal acceleration). If the Metal compiler isn't available, the build automatically falls back to CPU. To force a CPU-only build, pass `-DGPU_RUNTIME=CPU`.

:warning: You will probably get a *libc10.dylib can’t be opened because Apple cannot check it for malicious software* error on first run. Open **System Settings** and go to **Privacy & Security** and find the **Allow** button. You might need to repeat this several times until all torch libraries are loaded.

:warning: If you get a *Library not loaded: @rpath/libomp.dylib* error, try running `brew link libomp --force` before running OpenSplat.

## Docker Build

### CUDA

Navigate to the root directory of OpenSplat repo that has Dockerfile and run the following command to build the Docker image:

```bash
docker build -t opensplat .
```

The `-t` flag and other `--build-arg` let you tag and further customize your image across different ubuntu versions, CUDA/libtorch stacks, and hardware accelerators. 
For example, to build an image with Ubuntu 22.04, CUDA 12.1.1, libtorch 2.2.1, and support for CUDA architectures 7.0 and 7.5, run the following command:

```bash
docker build \
  -t opensplat:ubuntu-22.04-cuda-12.1.1-torch-2.2.1 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg TORCH_VERSION=2.2.1 \
  --build-arg CMAKE_CUDA_ARCHITECTURES="70;75;80" \
  --build-arg CMAKE_BUILD_TYPE=Release .
```

### ROCm via HIP

Navigate to the root directory of OpenSplat repo that has Dockerfile and run the following command to build the Docker image:
```bash
docker build \
  -t opensplat \
  -f Dockerfile.rocm .
```

The `-t` flag and other `--build-arg` let you tag and further customize your image across different ubuntu versions, CUDA/libtorch stacks, and hardware accelerators.
For example, to build an image with Ubuntu 22.04, CUDA 12.1.1, libtorch 2.2.1, ROCm 5.7.1, and support for ROCm architectures gfx906, run the following command:

```bash
docker build \
  -t opensplat:ubuntu-22.04-cuda-12.1.1-libtorch-2.2.1-rocm-5.7.1-llvm-16 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg TORCH_VERSION=2.2.1 \
  --build-arg ROCM_VERSION=5.7.1 \
  --build-arg PYTORCH_ROCM_ARCH="gfx906" \
  --build-arg CMAKE_BUILD_TYPE=Release .
```
Note: If you want to use ROCm 6.x, you need to switch to AMD version of pytorch docker as a base layer to build:
```bash
docker build \
  -t opensplat:ubuntu-22.04-libtorch-2.1.2-rocm-6.0.2 \
  -f Dockerfile.rocm6 .
```

## Run

To get started, download a dataset and extract it to a folder: [ [banana](https://drive.google.com/file/d/1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv/view?usp=sharing) ]  [ [truck](https://drive.google.com/file/d/1WWXo-GKo6d-yf-K1T1CswIdkdZrBNZ_e/view?usp=sharing) ] 

Then run from a command line prompt:

### Windows

```bash
cd c:\path\to\opensplat
opensplat.exe c:\path\to\banana -n 2000
```

### macOS / Linux

```bash
cd build
./opensplat /path/to/banana -n 2000
```

The program will generate an output `splat.ply` file which can then be dragged and dropped in one of the many [viewers](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#viewers) such as  https://playcanvas.com/viewer. You can also edit/cleanup the scene using https://playcanvas.com/supersplat/editor. The program will also output a `cameras.json` file in the same directory which can be used by some viewers.

To run on your own data, choose the path to an existing [COLMAP](https://colmap.github.io/), [OpenSfM](https://github.com/mapillary/OpenSfM), [ODX](https://github.com/WebODM/ODX) or [nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html) project. The project must have sparse points included (random initialization is not supported, see https://github.com/pierotofy/OpenSplat/issues/7).

There's several parameters you can tune. To view the full list:


```bash
./opensplat --help
```

### Google Colab

To run OpenSplat in Google Colab follow this [example notebook](https://colab.research.google.com/drive/1USqQsIBcqdOP6Fy0aVAyoXzTdpaEoTL_).

### Compression

To generate compressed splats (.splat files), use the `-o` option:

```bash
./opensplat /path/to/banana -o banana.splat
```

### Resume

You can resume training of a .PLY file by using the `--resume` option:

```bash
./opensplat /path/to/banana --resume ./splat.ply
```

### AMD GPU Notes

To train a model with AMD GPU using docker container, you can use the following command as a reference:
1. Launch the docker container with the following command:
```bash
docker run -it -v ~/data:/data --device=/dev/kfd --device=/dev/dri opensplat:ubuntu-22.04-libtorch-2.1.2-rocm-6.0.2 bash
```
2. Inside the docker container, run the following command to train the model:
```bash
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # AMD RX 6700 XT workaround 
cd /code/build
./opensplat /data/banana -n 2000
```
## Project Goals

We recently released OpenSplat, so there's lots of work to do.

 * Support for running on AMD cards (more testing needed)
 * Improve speed / reduce memory usage
 * Distributed computation using multiple machines
 * Real-time training viewer output
 * Automatic filtering
 * Your ideas?

 https://github.com/pierotofy/OpenSplat/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement

## Contributing

We welcome contributions! Pull requests are welcome.

## GPU Memory Notes

A single gaussian takes ~2000 bytes of memory, so currenly you need ~2GB of GPU memory for each million gaussians.

## Credits

The methods used in OpenSplat are originally based on [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html).

## License

The code in this repository, unless otherwise noted, is licensed under the AGPLv3.

The code from [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is originally licensed under the Apache 2.0 license and is © 2023 The Nerfstudio Team.
