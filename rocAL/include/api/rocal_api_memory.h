/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_MEMORY_H
#define MIVISIONX_ROCAL_API_MEMORY_H

#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - External Allocator API
 *
 * \defgroup group_rocal_memory API: AMD rocAL - Memory Management
 * \brief Allows external frameworks (PyTorch, JAX, TensorFlow) to provide
 * their own GPU memory allocator to avoid memory pool contention.
 */

/*! \brief Function pointer type for GPU memory allocation
 * \ingroup group_rocal_memory
 * \param [in] size Number of bytes to allocate
 * \param [in] stream HIP stream to associate allocation with (as void* for C compatibility)
 * \param [in] user_data Opaque pointer passed during registration
 * \return Pointer to allocated GPU memory, or nullptr on failure
 */
typedef void* (*RocalAllocFunc)(size_t size, void* stream, void* user_data);

/*! \brief Function pointer type for GPU memory deallocation
 * \ingroup group_rocal_memory
 * \param [in] ptr Pointer to GPU memory to free
 * \param [in] user_data Opaque pointer passed during registration
 */
typedef void (*RocalFreeFunc)(void* ptr, void* user_data);

/*!
 * \brief Register an external GPU memory allocator for a rocAL context
 * \ingroup group_rocal_memory
 *
 * When registered, rocAL will use the provided allocator functions
 * for all GPU memory allocations instead of hipMalloc/hipFree.
 * This enables integration with framework-specific memory pools
 * (e.g., PyTorch's HIPCachingAllocator) to avoid memory fragmentation
 * during training.
 *
 * \note This must be called BEFORE rocalVerify() and after rocalCreate().
 * \note The allocator is used for GPU memory only. CPU/pinned memory
 *       allocations are not affected.
 *
 * \param [in] context the rocAL context
 * \param [in] alloc_func allocation function (called instead of hipMalloc)
 * \param [in] free_func deallocation function (called instead of hipFree)
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
 * \ingroup group_rocal_memory
 *
 * \param [in] context the rocAL context
 * \return RocalStatus indicating success or failure
 */
extern "C" RocalStatus ROCAL_API_CALL rocalClearExternalAllocator(RocalContext context);

/*!
 * \brief Get the HIP stream used by rocAL for GPU operations
 * \ingroup group_rocal_memory
 *
 * Returns the primary HIP stream used by rocAL for GPU augmentation
 * operations. This can be used by external allocators that require
 * stream association for memory lifecycle tracking.
 *
 * \param [in] context the rocAL context
 * \return HIP stream as void*, or nullptr if CPU mode or invalid context
 */
extern "C" void* ROCAL_API_CALL rocalGetHipStream(RocalContext context);

#endif // MIVISIONX_ROCAL_API_MEMORY_H
