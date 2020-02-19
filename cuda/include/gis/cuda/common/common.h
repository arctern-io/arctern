#pragma once
#ifdef __CUDACC__
#define DEVICE_RUNNABLE __host__ __device__
#else
#define DEVICE_RUNNABLE
#endif

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>

#include <cmath>

#include "kernel_config.h"
