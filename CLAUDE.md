# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Prerequisites Setup (run once)
```bash
python rocAL-setup.py --backend HIP
```

### Build with HIP Backend (default)
```bash
mkdir build-hip && cd build-hip
cmake ../
make -j8
sudo make install
```

### Useful CMake Options
- `-D BACKEND=HIP|CPU` - Select GPU backend (default: HIP)
- `-D BUILD_PYPACKAGE=ON|OFF` - Build Python bindings (default: ON)
- `-D PYTHON_VERSION_SUGGESTED=3.x` - Specify Python version
- `-D AUDIO_SUPPORT=ON` - Enable audio features (default ON with ROCm > 6.2)
- `-D GPU_SUPPORT=OFF` - Build CPU-only

### Run Tests
```bash
cd build-hip
make test              # Run all tests
make test ARGS="-VV"   # Verbose output
```

### Verify Package Installation
```bash
mkdir rocAL-test && cd rocAL-test
cmake /opt/rocm/share/rocal/test/
ctest -VV
```

### Python Pybind Tests
```bash
mkdir rocal-pybind-test && cd rocal-pybind-test
cmake /opt/rocm/share/rocal/test/pybind
ctest -VV
```

## Architecture

rocAL is an AMD ROCm augmentation library for efficiently decoding and processing images, videos, and audio through a user-programmable processing graph.

### Core Pipeline Pattern
```cpp
RocalContext ctx = rocalCreate(batch_size, ROCAL_PROCESS_GPU, gpu_id);
// Add data loaders (readers + decoders)
// Add augmentation nodes to build the graph
rocalVerify(ctx);  // Validate the graph
while (/*has data*/) {
    rocalRun(ctx);  // Execute one iteration
    // Access output tensors
}
rocalRelease(ctx);
```

### Key Source Structure

**rocAL/include/api/** - Public C API headers:
- `rocal_api.h` - Main entry (rocalCreate, rocalVerify, rocalRun, rocalRelease)
- `rocal_api_data_loaders.h` - File/COCO/TF/Caffe/video readers
- `rocal_api_augmentation.h` - Image/video/audio augmentation operations
- `rocal_api_types.h` - Enums (RocalProcessMode, RocalTensorLayout, RocalDecoderType)

**rocAL/source/** - Implementation organized by module:
- `api/` - C API implementations
- `loaders/` - Data reader implementations  
- `decoders/` - JPEG (TurboJPEG/rocJPEG), video (FFMPEG/rocDecode), audio decoders
- `augmentations/` - Augmentation node implementations (color, geometry, effects, audio)
- `pipeline/` - Graph construction and execution
- `readers/` - File system, COCO, TFRecord, Caffe LMDB readers

**rocAL_pybind/** - Python bindings via pybind11

### Processing Backends
- **HIP** (default): GPU-accelerated via ROCm, uses amdclang++
- **CPU**: Host-only processing
- Decoder options: TurboJPEG (CPU), rocJPEG (GPU), FFMPEG (video), rocDecode (GPU video)

### Test Structure
- `tests/cpp_api/` - C++ unit/integration tests (basic_test, unit_tests, performance_tests, video_tests, etc.)
- `tests/python_api/` - Python reader and augmentation tests
- `tests/pybind/` - Python binding tests

Tests use sample data from `data/images/AMD-tinyDataSet/`.

## Environment Variables
```bash
export ROCM_PATH=/opt/rocm
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH  # For Python module
```

## Key Dependencies
- ROCm 6.4+ with HIP
- MIVisionX (AMD OpenVX)
- TurboJPEG, Protobuf, Half, RapidJSON
- Optional: rocDecode, rocJPEG, FFMPEG, libsndfile (audio)
