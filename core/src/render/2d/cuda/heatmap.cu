#include "render/2d/set_color.h"
#include "render/2d/cuda/heatmap.cuh"

#define TEMPLATE_GEN_PREFIX
#define T uint32_t
#include "render/2d/set_color.inl"

#define TEMPLATE_GEN_PREFIX
#define T float
#include "render/2d/set_color.inl"

#define TEMPLATE_GEN_PREFIX
#define T double
#include "render/2d/set_color.inl"


