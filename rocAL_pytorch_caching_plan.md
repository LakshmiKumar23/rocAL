# rocAL PyTorch CachingAllocator Integration Plan

## Overview

This document outlines the plan to integrate PyTorch's HIPCachingAllocator with rocAL to eliminate GPU memory contention during training workflows that use rocAL for data loading and augmentation with rocJPEG hardware decoding.

## Problem Statement

### Current Memory Architecture

rocAL and PyTorch maintain **separate GPU memory pools**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU Memory (16GB)                            │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │  hipMalloc pool         │  │  PyTorch CachingAllocator   │   │
│  │  (rocAL)                │  │  pool                       │   │
│  │                         │  │                             │   │
│  │  circular_buffer: 2GB   │  │  model weights: 2GB         │   │
│  │  ring_buffer: 1GB       │  │  activations: 4GB           │   │
│  │  rocjpeg: 512MB         │  │  gradients: 4GB             │   │
│  │  [freed but can't be    │  │  [freed memory cached here] │   │
│  │   used by PyTorch]      │  │                             │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
│         3.5GB                          10GB                      │
│                                                                  │
│  ❌ OOM when PyTorch needs 3GB more - rocAL's freed memory      │
│     is invisible to PyTorch's allocator                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Issues

1. **Memory Pool Fragmentation**: Each allocator manages its own pool. Memory freed by one cannot be reused by the other.

2. **OOM Despite Available Memory**: Training fails with OOM errors even when total free memory is sufficient, because it's fragmented across pools.

3. **Allocation Overhead**: rocAL uses raw `hipMalloc`/`hipFree` calls which are expensive OS-level operations (~50-200μs each).

4. **No Memory Visibility**: PyTorch's memory debugging tools (`torch.cuda.memory_stats()`, profilers) cannot see rocAL's allocations.

### Current Allocation Sites in rocAL

| File | Function | Purpose |
|------|----------|---------|
| `circular_buffer.cpp:162` | `init()` | Decoded image staging buffers |
| `ring_buffer.cpp:149-196` | `init()` | Output tensor buffers |
| `rocjpeg_decoder.cpp:216` | `decode_batch()` | Intermediate decode/resize buffers |
| `master_graph.cpp:635` | output allocation | Final output tensor buffer |

## Proposed Solution

### Unified Memory Pool

Route all rocAL GPU allocations through PyTorch's HIPCachingAllocator:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU Memory (16GB)                            │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           PyTorch CachingAllocator (single pool)          │  │
│  │                                                           │  │
│  │  rocAL circular_buffer: 2GB    ←── allocated via          │  │
│  │  rocAL ring_buffer: 1GB            raw_alloc_with_stream  │  │
│  │  rocAL rocjpeg: 512MB                                     │  │
│  │  model weights: 2GB                                       │  │
│  │  activations: 4GB                                         │  │
│  │  gradients: 4GB                                           │  │
│  │                                                           │  │
│  │  [all freed memory in ONE cache - reusable by anyone]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ✓ When rocAL frees rocjpeg buffer, PyTorch can reuse it        │
│  ✓ When PyTorch frees gradients, rocAL can reuse that memory    │
│  ✓ Allocator sees full picture - optimal block reuse            │
└─────────────────────────────────────────────────────────────────┘
```

### Stream Architecture

rocAL maintains its own streams for parallelism - this is preserved:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU                                       │
│                                                                  │
│  ┌──────────────────────────┐    ┌───────────────────────────┐  │
│  │     rocAL Pipeline       │    │    PyTorch Training       │  │
│  │                          │    │                           │  │
│  │  rocJPEG streams (N)     │    │   training stream         │  │
│  │      ↓ decode            │    │      ↓                    │  │
│  │  rocAL main stream       │    │   forward/backward        │  │
│  │      ↓ augment           │    │                           │  │
│  │  ring buffer             │───→│   input tensors           │  │
│  │                          │    │                           │  │
│  └──────────────────────────┘    └───────────────────────────┘  │
│         ↑                                   ↑                    │
│         │      PARALLEL EXECUTION           │                    │
│         └───────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key points:**
- rocJPEG keeps its multiple streams for maximum decode throughput from hardware
- rocAL main stream handles augmentation pipeline
- PyTorch training runs on its own stream
- Data loading and training execute in parallel (prefetch queue)
- Allocations use `raw_alloc_with_stream(size, rocal_stream)` for proper stream association

### Design Constraints

1. **No changes to rocAL core allocation infrastructure** - JAX and TensorFlow plugins must continue to work
2. **PyTorch as optional build-time dependency** - Only for rocAL_pybind, not core library
3. **Preserve rocJPEG parallelism** - Multiple decode streams for maximum hardware utilization
4. **Preserve data loading/training overlap** - rocAL and PyTorch on separate streams

## Implementation Plan

### Phase 1: External Allocator API in rocAL Core

#### New Header: `rocal_api_memory.h`

```cpp
#ifndef ROCAL_API_MEMORY_H
#define ROCAL_API_MEMORY_H

#include "rocal_api_types.h"

/*! \brief Function pointer type for GPU memory allocation
 *  \param size Number of bytes to allocate
 *  \param stream HIP stream to associate allocation with
 *  \param user_data Opaque pointer passed during registration
 *  \return Pointer to allocated memory, or nullptr on failure
 */
typedef void* (*RocalAllocFunc)(size_t size, void* stream, void* user_data);

/*! \brief Function pointer type for GPU memory deallocation
 *  \param ptr Pointer to memory to free
 *  \param user_data Opaque pointer passed during registration
 */
typedef void (*RocalFreeFunc)(void* ptr, void* user_data);

/*!
 * \brief Register an external GPU memory allocator for a rocAL context
 * \ingroup group_rocal
 *
 * When registered, rocAL will use the provided allocator functions
 * for all GPU memory allocations instead of hipMalloc/hipFree.
 * This must be called BEFORE rocalVerify().
 *
 * \param [in] context the rocAL context
 * \param [in] alloc_func allocation function
 * \param [in] free_func deallocation function
 * \param [in] user_data opaque pointer passed to alloc/free functions
 * \return RocalStatus indicating success or failure
 */
extern "C" RocalStatus ROCAL_API_CALL rocalSetExternalAllocator(
    RocalContext context,
    RocalAllocFunc alloc_func,
    RocalFreeFunc free_func,
    void* user_data);

/*!
 * \brief Clear the external allocator, reverting to default hipMalloc/hipFree
 * \ingroup group_rocal
 *
 * \param [in] context the rocAL context
 * \return RocalStatus indicating success or failure
 */
extern "C" RocalStatus ROCAL_API_CALL rocalClearExternalAllocator(RocalContext context);

/*!
 * \brief Get the HIP stream used by rocAL for GPU operations
 * \ingroup group_rocal
 *
 * \param [in] context the rocAL context
 * \return hipStream_t used by rocAL, or nullptr if CPU mode
 */
extern "C" void* ROCAL_API_CALL rocalGetHipStream(RocalContext context);

#endif // ROCAL_API_MEMORY_H
```

#### Internal Allocator Class: `hip_allocator.h`

```cpp
#pragma once

#include <functional>
#include <hip/hip_runtime.h>

class HipAllocator {
public:
    using AllocFunc = std::function<void*(size_t, hipStream_t)>;
    using FreeFunc = std::function<void(void*)>;

    HipAllocator() : _use_external(false), _stream(nullptr) {}

    void setExternalAllocator(AllocFunc alloc, FreeFunc free, hipStream_t stream) {
        _alloc_fn = alloc;
        _free_fn = free;
        _stream = stream;
        _use_external = true;
    }

    void clearExternalAllocator() {
        _use_external = false;
        _alloc_fn = nullptr;
        _free_fn = nullptr;
    }

    void* allocate(size_t size) {
        if (_use_external && _alloc_fn) {
            return _alloc_fn(size, _stream);
        }
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        if (_use_external && _free_fn) {
            _free_fn(ptr);
        } else {
            hipFree(ptr);
        }
    }

    bool isExternal() const { return _use_external; }

private:
    bool _use_external;
    AllocFunc _alloc_fn;
    FreeFunc _free_fn;
    hipStream_t _stream;
};
```

### Phase 2: Integrate Allocator into rocAL Components

#### Files to Modify

| File | Change |
|------|--------|
| `rocAL/include/pipeline/context.h` | Add `HipAllocator` member to context |
| `rocAL/source/api/rocal_api.cpp` | Implement `rocalSetExternalAllocator`, `rocalGetHipStream` |
| `rocAL/source/loaders/circular_buffer.cpp` | Use `context->allocator().allocate()` instead of `hipMalloc` |
| `rocAL/source/pipeline/ring_buffer.cpp` | Use `context->allocator().allocate()` instead of `hipMalloc` |
| `rocAL/source/decoders/image/rocjpeg_decoder.cpp` | Use allocator for intermediate buffers |
| `rocAL/source/pipeline/master_graph.cpp` | Use allocator for output tensor buffer |

#### Example Modification (circular_buffer.cpp)

```cpp
// Before
hipError_t err = hipMalloc((void **)&_dev_buffer[buffIdx], _output_mem_size);

// After
_dev_buffer[buffIdx] = _allocator->allocate(_output_mem_size);
if (!_dev_buffer[buffIdx]) {
    THROW("GPU allocation of size " + TOSTR(_output_mem_size) + " failed");
}
```

### Phase 3: PyTorch Integration in rocAL_pybind

#### CMakeLists.txt Addition

```cmake
# Optional PyTorch allocator integration
find_package(Torch QUIET)
if(Torch_FOUND)
    target_compile_definitions(${mod_tgt} PUBLIC ENABLE_PYTORCH_ALLOCATOR=1)
    target_include_directories(${mod_tgt} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${mod_tgt} PRIVATE ${TORCH_LIBRARIES})
    message("-- ${Green}PyTorch CachingAllocator integration: ENABLED${ColourReset}")
else()
    target_compile_definitions(${mod_tgt} PUBLIC ENABLE_PYTORCH_ALLOCATOR=0)
    message("-- ${Yellow}PyTorch CachingAllocator integration: DISABLED (Torch not found)${ColourReset}")
endif()
```

#### rocal_pybind.cpp Addition

```cpp
#if ENABLE_PYTORCH_ALLOCATOR
#include <c10/hip/HIPCachingAllocator.h>

static void* pytorch_alloc_wrapper(size_t size, void* stream, void* user_data) {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    auto& allocator = *c10::hip::HIPCachingAllocator::get();
    return allocator.raw_alloc_with_stream(size, hip_stream);
}

static void pytorch_free_wrapper(void* ptr, void* user_data) {
    if (ptr) {
        auto& allocator = *c10::hip::HIPCachingAllocator::get();
        allocator.raw_delete(ptr);
    }
}
#endif

// In PYBIND11_MODULE:
#if ENABLE_PYTORCH_ALLOCATOR
m.def("enablePyTorchAllocator", [](RocalContext context) {
    RocalStatus status = rocalSetExternalAllocator(
        context,
        pytorch_alloc_wrapper,
        pytorch_free_wrapper,
        nullptr  // user_data not needed
    );
    return status;
}, "Enable PyTorch HIPCachingAllocator for rocAL GPU allocations");

m.def("disablePyTorchAllocator", [](RocalContext context) {
    return rocalClearExternalAllocator(context);
}, "Disable PyTorch allocator, revert to hipMalloc");

m.def("isPyTorchAllocatorAvailable", []() {
    return true;
}, "Check if PyTorch allocator support was compiled in");
#else
m.def("isPyTorchAllocatorAvailable", []() {
    return false;
}, "Check if PyTorch allocator support was compiled in");
#endif
```

### Phase 4: Python Plugin Update

#### pytorch.py Modification

```python
class ROCALGenericIterator(object):
    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False,
                 multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0],
                 tensor_dtype=types.FLOAT, device="cpu", device_id=0,
                 display=False, use_pytorch_allocator=True):
        
        # ... existing init code ...
        
        self._using_pytorch_allocator = False
        if use_pytorch_allocator and self.device != "cpu":
            self._setup_pytorch_allocator()
    
    def _setup_pytorch_allocator(self):
        """Configure rocAL to use PyTorch's HIPCachingAllocator."""
        if not hasattr(b, 'isPyTorchAllocatorAvailable'):
            return
            
        if not b.isPyTorchAllocatorAvailable():
            import warnings
            warnings.warn(
                "rocAL was built without PyTorch allocator support. "
                "GPU memory will be managed separately, which may cause "
                "memory fragmentation during training."
            )
            return
        
        status = b.enablePyTorchAllocator(self.loader._handle)
        if status != 0:  # ROCAL_OK
            raise RuntimeError("Failed to enable PyTorch allocator for rocAL")
        
        self._using_pytorch_allocator = True
    
    def __del__(self):
        if self._using_pytorch_allocator:
            b.disablePyTorchAllocator(self.loader._handle)
        b.rocalRelease(self.loader._handle)
```

## Performance Benefits

### 1. Eliminates OOM-Induced Failures

Training runs to completion without crashes from memory fragmentation between pools.

### 2. Enables Larger Batch Sizes

With unified memory pool, memory is used more efficiently:

```python
# Before: Conservative batch size due to memory uncertainty
batch_size = 32  # Leave headroom for "the other allocator"

# After: Use what's actually available
batch_size = 64  # Full memory utilization
```

### 3. Reduces Memory Allocation Overhead

| Operation | hipMalloc (current) | CachingAllocator |
|-----------|--------------------|--------------------|
| Allocation | ~50-200μs (OS call) | ~1-5μs (cached block) |
| Deallocation | ~50-100μs | ~0.5μs (return to cache) |
| Frequency | Every batch | Rarely hits OS after warmup |

### 4. Better Memory Locality and Reuse

Memory blocks cycle efficiently between rocAL and PyTorch:

```
Training loop iteration:

Step 1: rocAL decodes batch → allocates 500MB
Step 2: rocAL augments → done, frees decode buffer  
Step 3: PyTorch forward pass needs 500MB for activations
        → Gets rocAL's just-freed block (cache hit)
        
Step 4: Backward pass frees activations
Step 5: rocAL next batch decode
        → Gets PyTorch's just-freed block
```

### 5. Enables Full Pipeline Overlap

```
Timeline (memory constrained - BEFORE):

rocAL:   [decode]──[wait for memory]──[augment]──[decode]──[wait]...
PyTorch: ────────[forward]────────────[backward]─────────[forward]...
                      ↑
              Memory contention causes stalls

Timeline (unified memory - AFTER):

rocAL:   [decode][augment][decode][augment][decode][augment]...
PyTorch: ────────[forward][backward][forward][backward]...
              ↑
         True parallel execution, prefetch queue always full
```

### 6. Quantified Impact (Expected)

| Metric | Improvement |
|--------|-------------|
| Memory efficiency | 15-30% more usable memory |
| Allocation latency | 10-100x faster (cached vs OS call) |
| Training throughput | 5-15% faster (reduced stalls) |
| Max batch size | 20-50% larger (better memory use) |
| OOM failures | Eliminated |

### 7. Enables Advanced PyTorch Features

With rocAL in PyTorch's memory ecosystem:

- **Gradient checkpointing**: Works correctly with memory pressure
- **torch.cuda.memory_stats()**: Shows accurate memory usage including rocAL
- **torch.cuda.empty_cache()**: Actually frees all memory when needed
- **Memory snapshots/profiling**: Full visibility into the pipeline

## Expected Performance Improvements by Configuration

The performance benefits vary depending on the decoder and backend configuration. The following are estimates to be validated after implementation.

### GPU Allocations by Decoder Type

**rocJPEG (Hardware Decoder)**

| Component | Allocation Location | Size (typical) |
|-----------|---------------------|----------------|
| `rocjpeg_decoder.cpp` intermediate buffers | GPU | batch × max_dim² × 3 |
| `circular_buffer.cpp` decode output | GPU | batch × max_dim² × 3 × depth |
| `ring_buffer.cpp` output tensors | GPU | batch × H × W × C × depth |
| Augmentation temporaries | GPU | varies |

**TurboJPEG (Software Decoder)**

| Component | Allocation Location | Size |
|-----------|---------------------|------|
| TurboJPEG decode buffers | CPU | - |
| `circular_buffer.cpp` with pinned memory | Host (pinned) | batch × max_dim² × 3 × depth |
| `ring_buffer.cpp` output tensors | GPU | batch × H × W × C × depth |
| Augmentation temporaries | GPU | varies |

### Expected Improvement by Configuration

| Scenario | Allocation Overhead Reduction | Memory Pool Unification | Overall Improvement |
|----------|------------------------------|------------------------|---------------------|
| rocJPEG + GPU augmentation | High (10-100x faster allocs) | High | **5-15%** |
| TurboJPEG + GPU augmentation | Moderate (fewer GPU allocs) | High | **2-5%** |
| TurboJPEG + CPU augmentation | Minimal | Low | **<1%** |

### Analysis

**rocJPEG (Hardware Decoder)**: Maximum benefit expected. Heavy GPU allocation activity from decode intermediate buffers, circular buffer in device memory, and augmentation. All allocations route through CachingAllocator, reducing allocation latency by 10-100x.

**TurboJPEG + GPU Backend**: Moderate benefit expected. Decode happens on CPU with pinned memory (not `hipMalloc`), so fewer GPU allocations. However:
- Ring buffer allocations still use GPU
- GPU augmentation buffers (crop, resize, normalize) still allocate on GPU
- Memory pool unification still prevents OOM from fragmentation
- Main bottleneck may be CPU decode, not GPU memory

**TurboJPEG + CPU Backend**: Minimal benefit. Most allocations are CPU-side. Integration provides little value in this configuration.

### Primary Value by Use Case

| Use Case | Primary Value |
|----------|---------------|
| rocJPEG training | Throughput improvement + memory stability |
| TurboJPEG + GPU training | Memory stability (OOM prevention) + modest throughput gain |
| Large batch sizes | Enables larger batches via better memory utilization |
| Long training runs | Eliminates OOM crashes from memory fragmentation |

*Note: These estimates should be validated with benchmarks after implementation.*

## Reference Implementation

This approach follows the pattern used by other PyTorch-integrated HIP/CUDA libraries:

- **gsplat**: https://github.com/ROCm/gsplat/blob/b01acd43e3c7fa942f95fda0974e9125e4de7395/gsplat/cuda/include/Common.cuh#L31

```cpp
// gsplat pattern
auto &caching_allocator = *::c10::hip::HIPCachingAllocator::get();
auto temp_storage = caching_allocator.allocate(temp_storage_bytes);
```

## Testing Plan

1. **Unit Tests**: Verify allocator callback mechanism works correctly
2. **Memory Tests**: Compare memory usage with/without integration
3. **Performance Tests**: Benchmark training throughput improvement
4. **Stress Tests**: Long-running training with memory pressure
5. **Compatibility Tests**: Ensure JAX/TensorFlow plugins unaffected

## Migration Guide

### For Users

**No changes required to existing training pipelines.** The PyTorch allocator integration is automatic and transparent.

Example training pipeline (unchanged):

```python
# dataloaders.py - existing code works as-is

def train_pipeline(data_path, batch_size, local_rank, world_size, ...):
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=local_rank,
                    rocal_cpu=False, output_memory_type=types.DEVICE_MEMORY, ...)
    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image_slice(jpegs, device='gpu', ...)
        # ... augmentations ...
        pipe.set_outputs(cmnp)
    return pipe

def get_rocal_train_loader(data_path, batch_size, local_rank, ...):
    pipe_train = train_pipeline(...)
    pipe_train.build()
    
    # This line unchanged - allocator integration happens internally
    train_loader = ROCALClassificationIterator(pipe_train, device="cuda", device_id=local_rank)
    
    prefetcher = Prefetcher(train_loader, rocal_cpu=False)
    return prefetcher, len(train_loader)
```

The following remain unchanged:
- `Pipeline` class and configuration
- `fn.decoders.image_slice` / `fn.decoders.image` with `device='gpu'`
- `output_memory_type = types.DEVICE_MEMORY`
- `Prefetcher` wrapper class
- Training loop code

### Explicit Control (Optional)

For debugging or comparison, users can explicitly disable the integration:

```python
# Opt-out of PyTorch allocator (uses legacy hipMalloc behavior)
train_loader = ROCALClassificationIterator(
    pipeline, 
    device="cuda",
    use_pytorch_allocator=False
)
```

Default is `use_pytorch_allocator=True` when built with PyTorch support.

### Build Requirements

To enable PyTorch allocator support, ensure PyTorch is installed before building rocAL:

```bash
pip install torch  # or install from ROCm PyTorch wheels
cd rocAL && mkdir build && cd build
cmake .. -DBUILD_PYPACKAGE=ON
make -j$(nproc)
```

The build system will automatically detect PyTorch and enable the integration.
