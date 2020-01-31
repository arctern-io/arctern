#include "window_cpu_2d.h"


namespace zilliz {
namespace render {


void WindowCPU2D::Init() {

    // Init an RGBA-mode context.
#if OSMESA_MAJOR_VERSION * 100 + OSMESA_MINOR_VERSION >= 305
    // Specify Z, stencil, accum sizes.
    context_ = OSMesaCreateContextExt(OSMESA_RGBA, 0, 0, 0, nullptr);
#else
    ctx = OSMesaCreateContext( OSMESA_RGBA, NULL );
#endif
    if (!context_) {
        // TODO: Add log here.
        printf("OSMesaCreateContext failed!\n");
        return;
    }

    GLsizei screen_width = (GLsizei) window_params().width();
    GLsizei screen_height = (GLsizei) window_params().height();

    // Init buffer for context.
    buffer_ = (GLubyte *) malloc(screen_width * screen_height * 4 * sizeof(GLubyte));

    // Bind the buffer to the context and make it current.
    if (!OSMesaMakeCurrent(context_, buffer_, GL_UNSIGNED_BYTE, screen_width, screen_height)) {
        // TODO: Add log here.
        printf("OSMesaMakeCurrent failed!\n");
        return;
    }
}


void WindowCPU2D::Terminate() {
    OSMesaDestroyContext(context_);
}


} // namespace render
} // namespace zilliz
