#include "window_gpu_2d.h"


namespace zilliz {
namespace render {


void WindowGPU2D::Init() {
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    GLint screen_width = (GLint) window_params().width();
    GLint screen_height = (GLint) window_params().height();

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, screen_width,
        EGL_HEIGHT, screen_height,
        EGL_NONE,
    };

    const EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE,
    };

    // 1. Initialize EGL
    auto &eglDpy = mutable_egl_dpy();
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;

    eglInitialize(eglDpy, &major, &minor);

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 3. Create a surface
    auto &eglSurf = mutable_egl_surf();
    eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg,
                                      pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);

    // 5. Create a context and make it current
    auto &eglCtx = mutable_egl_context();
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,
                              contextAttribs);

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

    glOrtho(0, screen_width, 0, screen_height, -1, 1);
}

void WindowGPU2D::Terminate() {
    eglMakeCurrent(mutable_egl_dpy(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(mutable_egl_dpy(), mutable_egl_context());
    eglDestroySurface(mutable_egl_dpy(), mutable_egl_surf());
    eglReleaseThread();
    eglTerminate(mutable_egl_dpy());
}

} // namespace render
} // namespace zilliz
