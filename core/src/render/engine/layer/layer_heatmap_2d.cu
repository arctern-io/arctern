#include "render/engine/common/memory.h"
#include "render/engine/layer/layer_heatmap_2d.h"
#include "render/utils/dataset/dataset_accessor.h"
#include "render/utils/color/color_parser.h"
#include <thrust/functional.h>
#include <exception>

namespace zilliz {
namespace render {
namespace engine {

const double eps = 1e-6;


void guassiankernel(double *kernel, int size, double sigma) {
    double sum = 0;
    double *data = kernel;

    for (int i = 0; i < size; ++i) {
        double index = (size >> 1) - i;
        if (size & 1)
            *(data + i) = exp(-(index * index) / (2 * sigma * sigma + eps));
        else {
            index -= 0.5;
            *(data + i) = exp(-(index * index) / (2 * sigma * sigma + eps));
        }
        sum += *(data + i);
    }

    for (int i = 0; i < size; ++i) {
        *(data + i) /= sum;
    }
}

void matproduct(double a[], double b[], double c[], int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

void guassiankernel2d(double *kernel, int sizeX, int sizeY, double sigmaX, double sigmaY) {
    double *matX = (double *) malloc(sizeX * sizeof(double));
    double *matY = (double *) malloc(sizeY * sizeof(double));
    guassiankernel(matX, sizeX, sigmaX);
    guassiankernel(matY, sizeY, sigmaY);
    matproduct(matX, matY, kernel, sizeX, 1, sizeY);
    free(matX);
    free(matY);
}

LayerHeatMap2D::LayerHeatMap2D()
    : num_vertices_(0) {
}

LayerHeatMap2D::~LayerHeatMap2D() {
    auto &mem_pool = MemManager::GetInstance().main_memory_pool();
    if (colors_ != nullptr) {
        mem_pool.Free(colors_);
    }
}

void LayerHeatMap2D::Init() {
    auto &data_params_type = plan_node_->data_param_type();
    switch (data_params_type[2]) {
        case ValueType::kValInt8:Interop<int8_t>();
            break;
        case ValueType::kValInt16:Interop<int16_t>();
            break;
        case ValueType::kValInt32:Interop<int32_t>();
            break;
        case ValueType::kValInt64:Interop<int64_t>();
            break;
        case ValueType::kValUInt8:Interop<u_int8_t>();
            break;
        case ValueType::kValUInt16:Interop<u_int16_t>();
            break;
        case ValueType::kValUInt32:Interop<u_int32_t>();
            break;
        case ValueType::kValUInt64:Interop<u_int64_t>();
            break;
        case ValueType::kValFloat:Interop<float>();
            break;
        case ValueType::kValDouble:Interop<double>();
            break;
        default:std::string msg = "cannot find value type";
//            THROW_RENDER_ENGINE_ERROR(VALUE_TYPE_NOT_FOUND, msg);
    }
}

void LayerHeatMap2D::Shader() {}

void
LayerHeatMap2D::Render() {}

__global__ void
HeatMapArray(double *count,
             double *dev_count,
             double *dev_kernel,
             int64_t kernel_size,
             int64_t width,
             int64_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int count_index = y * width + x;
    if (count[count_index] > 1e-8) {
        int r = kernel_size / 2;
        for (int m = -r; m <= r; m++) {
            if (x + m < 0 || x + m >= width)
                continue;
            for (int n = -r; n <= r; n++) {
                if (y + n < 0 || y + n >= height)
                    continue;
                int kernel_index = (r + n) * (2 * r + 1) + (m + r);
                int dev_index = (y + n) * width + (x + m);
                dev_count[dev_index] += count[count_index] * dev_kernel[kernel_index];
            }
        }
    }
}

template<typename T>
__global__ void SetCountValue(uint32_t *x,
                              uint32_t *y,
                              T *count,
                              double *pix_count,
                              int64_t num_vertices,
                              int64_t width,
                              int64_t height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < num_vertices; i += blockDim.x * gridDim.x) {
        uint32_t vertice_x = x[i];
        uint32_t vertice_y = y[i];
        int64_t index = vertice_y * width + vertice_x;
        if (index >= width * height)
            continue;
        pix_count[index] += count[i];
    }
}

__global__ void
mean_gpu_kernel(double *img_in,
                double *img_out,
                int64_t r,
                int64_t img_w,
                int64_t img_h) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    double gradient = 0.0;
    if (r > 10) r = 10;
    int count = 0;
    if ((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w)) {
        for (int m = -r; m <= r; m++) {
            if (row + m < 0 || row + m >= img_h)
                continue;
            for (int n = -r; n <= r; n++) {
                if (col + n < 0 || col + n >= img_w)
                    continue;
                int y = row + m;
                int x = col + n;
                gradient += img_in[y * img_w + x];
                count++;
            }
        }
        img_out[row * img_w + col] = gradient / count;
    }
}

template<typename T>
void
LayerHeatMap2D::Interop() {
    int dev_id = plan_node()->dev_id();
    CHECK_CUDA(cudaSetDevice(dev_id))

    auto &data_params = plan_node_->data_params();
    int64_t width = window_params_.width();
    int64_t height = window_params_.height();
    int64_t window_size = width * height;
    auto &mem_pool = MemManager::GetInstance().main_memory_pool();

    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto count_id = data_params[2];

    auto table_id = x_id;
    table_id.truncate_to_table_id();
    auto fragments_field = DatasetAccessor::GetFragments(input(), table_id);

    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kMeta);
        auto fragment_id = x_id;
        fragment_id.truncate_to_fragment_id();
        int64_t num_row = DatasetAccessor::GetNumRows(input(), fragment_id);
        num_vertices_ += num_row;
    }

    if (num_vertices_ <= 0) {
        colors_ = (unsigned char *) mem_pool.Malloc(window_size * 4 * sizeof(unsigned char));
        memset(colors_, 0, window_size * 4 * sizeof(unsigned char));
//        RENDER_ENGINE_LOG_INFO << "Failed, because num of vertices <= 0";
        return;
    }

    T *count = nullptr;
    double *pix_count;

    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        y_id.set_fragment_field(fragments_field[i]);
        count_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        count_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);

        vertices_x_ = dataset_accessor()->GetChunkDataForHeatmap<uint32_t>(x_id, dev_id);
        vertices_y_ = dataset_accessor()->GetChunkDataForHeatmap<uint32_t>(y_id, dev_id);
        count = dataset_accessor()->GetChunkDataForHeatmap<T>(count_id, dev_id);
    }

    cuda_state = cudaMalloc((void **) &pix_count, window_size * sizeof(double));
    cuda_state = cudaMemset(pix_count, 0, window_size * sizeof(double));
    SetCountValue<T> << < 256, 1024 >>
        > (vertices_x_, vertices_y_, count, pix_count, num_vertices_, width, height);

    float max_pix = 0.0f;
    float scale = plan_node()->map_scale_ratio() * 0.4;
    int d = pow(2, scale);
    int64_t kernel_size = d * 2 + 3;

    double *kernel = (double *) malloc(kernel_size * kernel_size * sizeof(double));
    guassiankernel2d(kernel, kernel_size, kernel_size, kernel_size, kernel_size);
    double *dev_kernel;
    cuda_state = cudaMalloc((void **) &dev_kernel, kernel_size * kernel_size * sizeof(double));
    cudaMemcpy(dev_kernel, kernel, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);
    double *dev_count;
    cuda_state = cudaMalloc((void **) &dev_count, window_size * sizeof(double));
    cuda_state = cudaMemset(dev_count, 0, window_size * sizeof(double));

    const unsigned int blockW = 32;
    const unsigned int blockH = 32;
    const dim3 threadBlock(blockW, blockH);
    const dim3 grid(iDivUp(width, blockW), iDivUp(height, blockH));
    HeatMapArray << < grid, threadBlock >>
        > (pix_count, dev_count, dev_kernel, kernel_size, width, height);

    double *color_count;
    cuda_state = cudaMalloc((void **) &color_count, window_size * sizeof(double));
    cuda_state = cudaMemset(color_count, 0, window_size * sizeof(double));
    int64_t mean_radius = (int) (log((kernel_size - 3) / 2) / 0.4);

    mean_gpu_kernel << < grid, threadBlock >>
        > (dev_count, color_count, mean_radius + 1, width, height);
    mean_gpu_kernel << < grid, threadBlock >>
        > (color_count, dev_count, mean_radius / 2 + 1, width, height);

    auto host_count = (double *) mem_pool.Malloc(window_size * sizeof(double));
    cudaMemcpy(host_count, dev_count, window_size * sizeof(double), cudaMemcpyDeviceToHost);
    for (auto k = 0; k < window_size; k++) {
        if (max_pix < host_count[k])
            max_pix = host_count[k];
    }

    color_gradient_.createDefaultHeatMapGradient();
    colors_ = (unsigned char *) mem_pool.Malloc(window_size * 4 * sizeof(unsigned char));

    int64_t c_offset = 0;
    for (auto j = 0; j < window_size; j++) {
        double value = host_count[j] / max_pix;
        double color_r, color_g, color_b;
        color_gradient_.getColorAtValue(value, color_r, color_g, color_b);
        int color_ir = color_r * 255.0;
        int color_ig = color_g * 255.0;
        int color_ib = color_b * 255.0;
        int color_ia = value * 255.0;
        colors_[c_offset++] = (unsigned char) ((color_ir & 0x000000ff));
        colors_[c_offset++] = (unsigned char) ((color_ig & 0x000000ff));
        colors_[c_offset++] = (unsigned char) ((color_ib & 0x000000ff));
        colors_[c_offset++] = (unsigned char) ((color_ia & 0x000000ff));
    }
    free(kernel);
    free(host_count);
    CHECK_CUDA(cudaFree(pix_count))
    CHECK_CUDA(cudaFree(dev_kernel))
    CHECK_CUDA(cudaFree(dev_count))
    CHECK_CUDA(cudaFree(color_count))
}

} // namespace engine
} // namespace render
} // namespace zilliz
