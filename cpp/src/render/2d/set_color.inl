#include "render/2d/set_color.h"
#ifndef TEMPLATE_GEN_PREFIX
// dummy, to make IDE happy
#define TEMPLATE_GEN_PREFIX extern
#endif

#ifndef T
// dummy, to make IDE happy
#define T int
#endif

namespace zilliz {
namespace render {

TEMPLATE_GEN_PREFIX template void set_colors_gpu<T>(float *colors,
                                                 uint32_t* input_x,
                                                 uint32_t* input_y,
                                                 T* input_c,
                                                 int64_t num,
                                                 VegaHeatMap &vega_heat_map);
} //render
} // namespace zilliz

#undef T
#undef TEMPLATE_GEN_PREFIX
