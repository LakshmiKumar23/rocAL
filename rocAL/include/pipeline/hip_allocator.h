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

#pragma once

#include <cstddef>
#include "rocal_api_memory.h"

#if ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

class HipAllocator {
   public:
    HipAllocator() : _use_external(false), _alloc_fn(nullptr), _free_fn(nullptr), _user_data(nullptr), _stream(nullptr) {}

    void setExternalAllocator(RocalAllocFunc alloc_fn, RocalFreeFunc free_fn, void* user_data) {
        _alloc_fn = alloc_fn;
        _free_fn = free_fn;
        _user_data = user_data;
        _use_external = (alloc_fn != nullptr && free_fn != nullptr);
    }

    void setStream(void* stream) {
        _stream = stream;
    }

    void clearExternalAllocator() {
        _use_external = false;
        _alloc_fn = nullptr;
        _free_fn = nullptr;
        _user_data = nullptr;
    }

    void* allocate(size_t size) {
        if (size == 0) return nullptr;

        if (_use_external && _alloc_fn) {
            return _alloc_fn(size, _stream, _user_data);
        }

#if ENABLE_HIP
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
            return nullptr;
        }
        return ptr;
#else
        return nullptr;
#endif
    }

    void deallocate(void* ptr) {
        if (!ptr) return;

        if (_use_external && _free_fn) {
            _free_fn(ptr, _user_data);
            return;
        }

#if ENABLE_HIP
        (void)hipFree(ptr);
#endif
    }

    bool isExternal() const { return _use_external; }

   private:
    bool _use_external;
    RocalAllocFunc _alloc_fn;
    RocalFreeFunc _free_fn;
    void* _user_data;
    void* _stream;
};
