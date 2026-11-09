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

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/macro.h"
#include <driver_types.h>
#include <regex>

namespace turbomind {

void ReportCudaError(cudaError_t ec, const char* file, int line)
{
    core::Scope _("TM_CUDA_CHECK", file, line);
    core::Logger::Instance().LogFatalImpl(file, line, std::format("CUDA error: {}", cudaGetErrorString(ec)));
}

void ReportCuDrvError(CUresult ec, const char* file, int line)
{
    core::Scope _("TM_CUDRV_CHECK", file, line);
    const char* str{};
    cuGetErrorString(ec, &str);
    core::Logger::Instance().LogFatalImpl(file, line, std::format("CUDA Driver error: {}", str ? str : "Unknown"));
}

/* ***************************** common utils ****************************** */

int getSMVersion()
{
    int device{-1};
    TM_CUDA_CHECK(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    TM_CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    TM_CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

int getSMCount()
{
    int device{-1};
    TM_CUDA_CHECK(cudaGetDevice(&device));
    int sm_count{};
    TM_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    return sm_count;
}

/* ************************** end of common utils ************************** */
}  // namespace turbomind
