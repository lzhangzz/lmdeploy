/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublasLt.h>
#include <cublas_v2.h>
#ifdef SPARSITY_ENABLED
#include <cusparseLt.h>
#endif

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/cuda_bf16_wrapper.h"

namespace turbomind {

/* **************************** debug tools ********************************* */

// --- Unified CUDA error checking ---

void ReportCudaError(cudaError_t ec, const char* file, int line);
void ReportCuDrvError(CUresult ec, const char* file, int line);

#define TM_CUDA_CHECK(expr)                                                                                            \
    do {                                                                                                               \
        if (auto _ec = (expr); TM_UNLIKELY(_ec != cudaSuccess)) {                                                      \
            ::turbomind::ReportCudaError(_ec, __FILE__, __LINE__);                                                     \
        }                                                                                                              \
    } while (0)

#define TM_CUDRV_CHECK(expr)                                                                                           \
    do {                                                                                                               \
        if (auto _ec = (expr); TM_UNLIKELY(_ec != CUDA_SUCCESS)) {                                                     \
            ::turbomind::ReportCuDrvError(_ec, __FILE__, __LINE__);                                                    \
        }                                                                                                              \
    } while (0)

/* ***************************** common utils ****************************** */

int getSMVersion();

int getSMCount();

class CudaDeviceGuard {
public:
    CudaDeviceGuard(int device)
    {
        TM_CUDA_CHECK(cudaGetDevice(&last_device_id_));
        if (device != last_device_id_) {
            TM_CUDA_CHECK(cudaSetDevice(device));
        }
    }

    ~CudaDeviceGuard()
    {
        TM_CUDA_CHECK(cudaSetDevice(last_device_id_));
    }

private:
    int last_device_id_{-1};
};

/* ************************** end of common utils ************************** */
}  // namespace turbomind
