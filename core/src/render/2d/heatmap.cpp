#include <iostream>
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "heatmap.h"
#include "render/utils/color/color_gradient.h"

namespace zilliz {
namespace render {

const double eps = 1e-6;

void
guassiankernel(float *kernel, int size, float sigma) {
    float sum = 0;
    float *data = kernel;

    for (int i = 0; i < size; ++i) {
        float index = (size >> 1) - i;
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

void
matproduct(float a[], float b[], float c[], int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

void
guassiankernel2d(float *kernel, int sizeX, int sizeY, float sigmaX, float sigmaY) {
    float *matX = (float *) malloc(sizeX * sizeof(float));
    float *matY = (float *) malloc(sizeY * sizeof(float));
    guassiankernel(matX, sizeX, sigmaX);
    guassiankernel(matY, sizeY, sigmaY);
    matproduct(matX, matY, kernel, sizeX, 1, sizeY);
    free(matX);
    free(matY);
}

template<typename T>
void SetCountValue_cpu(float* out, uint32_t* in_x, uint32_t* in_y, T* in_c, int64_t num, int64_t width, int64_t height) {
    for (int i = 0; i < num; i++) {
        uint32_t vertice_x = in_x[i];
        uint32_t vertice_y = in_y[i];
        int64_t index = vertice_y * width + vertice_x;
        if (index >= width * height)
            continue;
        out[index] += in_c[i];
    }
}

void
HeatMapArray_cpu(float *in_count, float *out_count, float *kernel, int64_t kernel_size, int64_t width, int64_t height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int count_index = y * width + x;
            if (in_count[count_index] > 1e-8) {
                int r = kernel_size / 2;
                for (int m = -r; m <= r; m++) {
                    if (x + m < 0 || x + m >= width)
                        continue;
                    for (int n = -r; n <= r; n++) {
                        if (y + n < 0 || y + n >= height)
                            continue;
                        int kernel_index = (r + n) * (2 * r + 1) + (m + r);
                        int dev_index = (y + n) * width + (x + m);
                        out_count[dev_index] += in_count[count_index] * kernel[kernel_index];
                    }
                }
            }
        }
    }
}

void
MeanKernel_cpu(float *img_in, float *img_out, int64_t r, int64_t img_w, int64_t img_h) {
    for (int row = 0; row < img_h; row++) {
        for (int col = 0; col < img_w; col++) {
            float gradient = 0.0;
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
    }
}

template<typename T>
HeatMap<T>::HeatMap()
    : vertices_x_(nullptr), vertices_y_(nullptr), num_vertices_(0) {
}

template<typename T>
HeatMap<T>::HeatMap(std::shared_ptr<uint32_t> input_x, std::shared_ptr<uint32_t > input_y, std::shared_ptr<uint32_t > count, int64_t num_vertices)
    : vertices_x_(input_x), vertices_y_(input_y), count_(count), num_vertices_(num_vertices) {
}

template<typename T> void
HeatMap<T>::InputInit() {
   array_vector_ = input().array_vector;
   VegaHeatMap vega_heatmap(input().vega);
   heatmap_vega_ = vega_heatmap;
}


template<typename T> void
HeatMap<T>::DataInit() {
    set_colors();

    WindowParams window_params = heatmap_vega_.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    uint32_t* input_x = (uint32_t *) malloc( window_size* sizeof(uint32_t));
    uint32_t* input_y = (uint32_t *) malloc(window_size * sizeof(uint32_t));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input_x[i * width + j] = j;
            input_y[i * width + j] = height - i;
        }
    }
    vertices_x_ = std::shared_ptr<uint32_t >(input_x);
    vertices_y_ = std::shared_ptr<uint32_t >(input_y);
}

template<typename T>
void HeatMap<T>::set_colors_cpu() {
    WindowParams window_params = heatmap_vega_.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    float* pix_count = (float *) malloc(window_size * sizeof(float));
    memset(pix_count, 0, window_size * sizeof(float));
    SetCountValue_cpu<T>(pix_count, vertices_x_.get(), vertices_y_.get(), count_.get(), num_vertices_, width, height);

    double scale = heatmap_vega_.map_scale() * 0.4;
    int d = pow(2, scale);
    int64_t kernel_size = d * 2 + 3;

    float *kernel = (float *) malloc(kernel_size * kernel_size * sizeof(float));
    guassiankernel2d(kernel, kernel_size, kernel_size, kernel_size, kernel_size);

    float *heat_count = (float *) malloc(window_size * sizeof(float));
    HeatMapArray_cpu(pix_count, heat_count, kernel, kernel_size, width, height);

    float* color_count = (float *) malloc(window_size * sizeof(float));
    int64_t mean_radius = (int) (log((kernel_size - 3) / 2) / 0.4);
    MeanKernel_cpu(color_count, heat_count, mean_radius + 1, width, height);
    MeanKernel_cpu(heat_count, color_count, mean_radius / 2 + 1, width, height);

    float max_pix = 0;
    for (auto k = 0; k < window_size; k++) {
        if (max_pix < heat_count[k])
            max_pix = heat_count[k];
    }
    ColorGradient color_gradient;
    color_gradient.createDefaultHeatMapGradient();
    colors_ = (float *) malloc(window_size * 4 * sizeof(float));

    int64_t c_offset = 0;
    for (auto j = 0; j < window_size; j++) {
        float value = heat_count[j] / max_pix;
        float color_r, color_g, color_b;
        color_gradient.getColorAtValue(value, color_r, color_g, color_b);
        colors_[c_offset++] = color_r;
        colors_[c_offset++] = color_g;
        colors_[c_offset++] = color_b;
        colors_[c_offset++] = value;
    }

    free(pix_count);
    free(kernel);
    free(heat_count);
    free(color_count);
}

template<typename T>
void HeatMap<T>::Draw() {
    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    glDrawArrays(GL_POINTS, 0, num_vertices_);
    glFlush();

    glDeleteVertexArrays(1, &VAO_);
    glDeleteBuffers(2, VBO_);
}

template<typename T>
void HeatMap<T>::Shader() {
    const char *vertexShaderSource = \
        "#version 430 core\n"
        "layout (location = 0) in uint posX;\n"
        "layout (location = 1) in uint posY;\n"
        "layout (location = 2) in vec4 point_color;\n"
        "layout (location = 3) uniform vec2 screen_info;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   float tmp_x = posX;\n"
        "   float tmp_y = posY;\n"
        "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / screen_info.y) - 1, 0, 1);\n"
        "   color=point_color;\n"
        "}";

    const char *fragmentShaderSource = \
        "#version 430 core\n"
        "in vec4 color;\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = color;\n"
        "}";


    int success;
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::cout << "vertex shader compile failed.";
    }
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::cout << "fragment shader compile failed.";
    }
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        std::cout << "shader program link failed.";
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(1, &VAO_);
    glGenBuffers(3, VBO_);

    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_.get(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_.get(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), colors_, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO_ );
    glBindVertexArray(VAO_);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 1);
    glBindVertexArray(1);

    glUseProgram(shaderProgram);
    glUniform2f(3, window()->window_params().width(), window()->window_params().height());
    glBindVertexArray(VAO_);
}

template<typename T> std::shared_ptr<uint8_t >
HeatMap<T>::Render(){
//    InputInit();
    WindowsInit(heatmap_vega_.window_params());
//    DataInit();
    Shader();
    Draw();
    Finalize();
    return Output();
}

} //namespace render
} //namespace zilliz
