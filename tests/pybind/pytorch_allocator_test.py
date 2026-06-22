#!/usr/bin/env python3
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Test to compare GPU memory usage with and without PyTorch CachingAllocator integration

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types


def get_gpu_memory_stats():
    """Get current GPU memory statistics from PyTorch."""
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def reset_memory_stats():
    """Reset peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


class SimpleModel(nn.Module):
    """Simple CNN for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def create_pipeline(data_path, batch_size, crop_size, device_id):
    """Create a rocAL pipeline with augmentations."""
    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=4,
        device_id=device_id,
        rocal_cpu=False,  # GPU backend
        prefetch_queue_depth=2,
    )
    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path)
        images = fn.decoders.image(jpegs, output_type=types.RGB, file_root=data_path)
        images = fn.resize(images, resize_width=crop_size, resize_height=crop_size)
        images = fn.color_twist(images, brightness=0.2, contrast=0.2, saturation=0.2)
        images = fn.crop_mirror_normalize(
            images,
            crop=[crop_size, crop_size],
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(),
            output_layout=types.NCHW,
            output_dtype=types.FLOAT,
        )
        pipe.set_outputs(images)
    pipe.build()
    return pipe


def run_training_iterations(data_path, batch_size, crop_size, num_iterations,
                            use_pytorch_allocator, device_id):
    """Run training iterations and return memory stats."""
    device = torch.device(f"cuda:{device_id}")
    reset_memory_stats()

    # Create model and optimizer
    model = SimpleModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create rocAL pipeline and iterator
    pipe = create_pipeline(data_path, batch_size, crop_size, device_id)
    iterator = ROCALClassificationIterator(
        pipe,
        device=device,
        use_pytorch_allocator=use_pytorch_allocator,
    )

    print(f"  Running {num_iterations} iterations with use_pytorch_allocator={use_pytorch_allocator}")

    # Training loop
    model.train()
    for i, (images, labels) in enumerate(iterator):
        if i >= num_iterations:
            break

        # images may be a list of tensors; stack them into a batch
        if isinstance(images, list):
            images = torch.stack(images)
        # Remove extra leading dimension if present (e.g., [1, N, C, H, W] -> [N, C, H, W])
        if images.dim() == 5 and images.size(0) == 1:
            images = images.squeeze(0)
        # Forward pass
        outputs = model(images)
        # Use random labels since we just need memory measurement
        fake_labels = torch.randint(0, 10, (images.size(0),), device=device)
        loss = criterion(outputs, fake_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            stats = get_gpu_memory_stats()
            print(f"    Iteration {i+1}: allocated={stats['allocated_mb']:.1f}MB, "
                  f"reserved={stats['reserved_mb']:.1f}MB")

    # Get final stats
    final_stats = get_gpu_memory_stats()

    # Cleanup
    del iterator
    del pipe
    del model
    del optimizer
    torch.cuda.empty_cache()

    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test GPU memory usage with/without PyTorch CachingAllocator integration"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet",
        help="Path to image dataset",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--crop-size", type=int, default=224, help="Crop size")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    # Check if PyTorch allocator is available
    try:
        import rocal_pybind
        allocator_available = rocal_pybind.isPyTorchAllocatorAvailable()
    except AttributeError:
        allocator_available = False

    print("=" * 70)
    print("PyTorch CachingAllocator Integration Test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/HIP available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(args.device_id)}")
    print(f"PyTorch allocator available in rocAL: {allocator_available}")
    print(f"Dataset: {args.data_path}")
    print(f"Batch size: {args.batch_size}, Crop size: {args.crop_size}")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)

    if not allocator_available:
        print("\nWARNING: PyTorch allocator not available in rocAL build.")
        print("Rebuild rocAL_pybind with PyTorch support to enable comparison.")
        print("Running with use_pytorch_allocator=False only.\n")

    # Test WITHOUT PyTorch allocator
    print("\n[Test 1] Running WITHOUT PyTorch CachingAllocator integration...")
    stats_without = run_training_iterations(
        args.data_path, args.batch_size, args.crop_size,
        args.iterations, use_pytorch_allocator=False, device_id=args.device_id
    )

    # Give some time for memory to settle
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Test WITH PyTorch allocator (if available)
    if allocator_available:
        print("\n[Test 2] Running WITH PyTorch CachingAllocator integration...")
        stats_with = run_training_iterations(
            args.data_path, args.batch_size, args.crop_size,
            args.iterations, use_pytorch_allocator=True, device_id=args.device_id
        )

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\nWithout PyTorch Allocator:")
    print(f"  Peak allocated: {stats_without['max_allocated_mb']:.1f} MB")
    print(f"  Peak reserved:  {stats_without['max_reserved_mb']:.1f} MB")

    if allocator_available:
        print("\nWith PyTorch Allocator:")
        print(f"  Peak allocated: {stats_with['max_allocated_mb']:.1f} MB")
        print(f"  Peak reserved:  {stats_with['max_reserved_mb']:.1f} MB")

        # Calculate savings
        reserved_diff = stats_without['max_reserved_mb'] - stats_with['max_reserved_mb']
        reserved_pct = (reserved_diff / stats_without['max_reserved_mb']) * 100 if stats_without['max_reserved_mb'] > 0 else 0

        print("\nMemory Savings with Unified Allocator:")
        print(f"  Reserved memory reduction: {reserved_diff:.1f} MB ({reserved_pct:.1f}%)")

        if reserved_diff > 0:
            print("\n  The unified allocator reduces memory fragmentation by using")
            print("  a single memory pool for both rocAL and PyTorch allocations.")
        else:
            print("\n  NOTE: Memory savings may vary depending on workload and GPU.")
            print("  The benefit is most visible with larger batch sizes and longer runs.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
