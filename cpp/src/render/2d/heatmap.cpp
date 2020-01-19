#include <iostream>
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "render/2d/heatmap.h"

namespace zilliz {
namespace render {

template
class HeatMap<float>;

template
class HeatMap<double>;

template
class HeatMap<uint32_t>;

template<typename T>
HeatMap<T>::HeatMap()
    : vertices_x_(nullptr), vertices_y_(nullptr), num_vertices_(0) {
}

template<typename T>
HeatMap<T>::HeatMap(uint32_t* input_x,
                    uint32_t* input_y,
                    T* count,
                    int64_t num_vertices)
    : vertices_x_(input_x), vertices_y_(input_y), count_(count), num_vertices_(num_vertices) {
}

template<typename T>
void
HeatMap<T>::InputInit() {
    array_vector_ = input().array_vector;
    VegaHeatMap vega_heatmap(input().vega);
    heatmap_vega_ = vega_heatmap;
}

template<typename T>
void
HeatMap<T>::DataInit() {
    WindowParams window_params = heatmap_vega_.window_params();
    int64_t width = window_params.width();
    int64_t height = window_params.height();
    int64_t window_size = width * height;

    colors_ = (float *) malloc(window_size * 4 * sizeof(float));
    set_colors(colors_, vertices_x_, vertices_y_, count_, num_vertices_, heatmap_vega_);

    vertices_x_ = (uint32_t *) malloc(window_size * sizeof(uint32_t));
    vertices_y_ = (uint32_t *) malloc(window_size * sizeof(uint32_t));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            vertices_x_[i * width + j] = j;
            vertices_y_[i * width + j] = height - i - 1;
        }
    }

    num_vertices_ = window_size;
}

template<typename T>
void HeatMap<T>::Draw() {
//    glEnable(GL_PROGRAM_POINT_SIZE);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

#ifdef USE_GPU
    glDrawArrays(GL_POINTS, 0, num_vertices_);
    glFlush();

    glDeleteVertexArrays(1, &VAO_);
    glDeleteBuffers(2, VBO_);
#else
    glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1, 1);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    int offset = 0;
    std::vector<int32_t> vertices(num_vertices_ * 2);
    for (auto i = 0; i < num_vertices_; i++) {
        vertices[offset++] = vertices_x_[i];
        vertices[offset++] = vertices_y_[i];
    }
    glColorPointer(4, GL_FLOAT, 0, colors_);
    glVertexPointer(2, GL_INT, 0, &vertices[0]);

    glDrawArrays(GL_POINTS, 0, num_vertices_);
    glFlush();
#endif
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
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), colors_, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO_);
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

template<typename T>
std::shared_ptr<uint8_t>
HeatMap<T>::Render() {
//    InputInit();
    WindowsInit(heatmap_vega_.window_params());
    DataInit();
#ifdef USE_GPU
    Shader();
#endif
    Draw();
    Finalize();
    return Output();
}

} //namespace render
} //namespace zilliz
