#pragma once
#ifdef __CUDACC__
#define DEVICE_RUNNABLE __host__ __device__
#else
#define DEVICE_RUNNABLE
#endif

#include "kernel_config.h"
