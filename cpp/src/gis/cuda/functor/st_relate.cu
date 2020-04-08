#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_relate.h"
namespace arctern {
namespace gis {
namespace cuda {
void ST_Relate(const GeometryVector& left_vec, const GeometryVector& right_vec,
               de9im::Matrix input_matrix,
               de9im::Matrix* host_output_matrixes) {
  assert(left_vec.size() == right_vec.size());
  auto size = left_vec.size();

}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
