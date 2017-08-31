#include "screen.hh"
#include "ogl.hh"
#include <GL/glx.h>
#include <CL/cl.hpp>

struct process_params {
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Kernel kernel;
  cl::ImageGL tex;
} params;
cl::Buffer cl_spheres;
cl::BufferGL cl_vbo;
std::vector<cl::Memory> cl_vbos;
GLuint vbo;

struct Sphere {
  cl_float radius;
  cl_float dummy1;
  cl_float dummy2;
  cl_float dummy3;
  cl_float3 position;
  cl_float3 color;
  cl_float3 emission;
  Sphere(cl_float n_radius, cl_float3 n_position, cl_float3 n_color
      , cl_float3 n_emission)
    : radius(n_radius)
    , position(n_position)
    , color(n_color)
    , emission(n_emission) {
  }
};

#define _float3(x, y, z) {{ x, y, z }} // macro to replace ugly initializer braces
Sphere cpu_spheres[] = {
  Sphere(200.f, _float3(-200.6f, 0.0f, 0.0f),   _float3(0.75f, 0.25f, 0.25f), _float3(0, 0, 0)),
  Sphere(200.f, _float3(200.6f, 0.0f, 0.0f),    _float3(0.25f, 0.25f, 0.75f), _float3(0, 0, 0)),
  Sphere(200.f, _float3(0.0f, -200.4f, 0.0f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(200.f, _float3(0.0f, 200.4f, 0.0f),    _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(200.f, _float3(0.0f, 0.0f, -200.4f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(200.f, _float3(0.0f, 0.0f, 202.0f),    _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(0.16f, _float3(-0.25f, -0.24f, -0.1f), _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(0.16f, _float3(0.25f, -0.24f, 0.1f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0, 0, 0)),
  Sphere(  1.f, _float3(0.0f, 1.36f, 0.0f),     _float3(0.0f, 0.0f, 0.0f),    _float3(9.0f, 8.0f, 6.0f))
};
const int num_spheres = sizeof(cpu_spheres) / sizeof(Sphere);

static const float proj_matrix[16] = {
  1.f, 0.f, 0.f, 0.f,
  0.f, 1.f, 0.f, 0.f,
  0.f, 0.f, 1.f, 0.f,
  0.f, 0.f, 0.f, 1.f
};

screen *g_screen = new screen("bblik", 800, 600);

void check_clgl_interop_availiability(const cl::Device &device) {
#if defined (__APPLE__) || defined(MACOSX)
  std::string cl_gl_sharing_ext_name = "cl_APPLE_gl_sharing";
#else
  std::string cl_gl_sharing_ext_name = "cl_khr_gl_sharing";
#endif
  std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
  std::size_t found = extensions.find(cl_gl_sharing_ext_name); // TODO
  if (found == std::string::npos)
    die("device \"%s\" does not support OpenGL-OpenCL interoperability"
        , device.getInfo<CL_DEVICE_NAME>().c_str());
}

void load() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform platform = platforms[0];

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  params.device = devices[0];
  check_clgl_interop_availiability(params.device);

  cl_context_properties properties[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
    0
  };

  params.context = cl::Context(params.device, properties);

  params.queue = cl::CommandQueue(params.context, params.device);

  std::string source = read_file_to_string("opencl_kernel.cl");
  const char *source_c_str = source.c_str();
  params.program = cl::Program(params.context, source_c_str);
  // "-cl-fast-relaxed-math"
  cl_int result = params.program.build({ params.device });
  if (result) {
    if (result == CL_BUILD_PROGRAM_FAILURE) {
      std::string build_log
        = params.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(params.device);
      printf("Build log:\n%s\n", build_log.c_str());
    }
    die("Failed to compile OpenCL program (%d)", result);
  }

  params.kernel = cl::Kernel(params.program, "render_kernel");

  cl_spheres = cl::Buffer(params.context, CL_MEM_READ_ONLY
      , num_spheres * sizeof(Sphere));

  // create opengl stuff
  glClearColor(0.2, 0.2, 0.2, 1.0);

  glMatrixMode(GL_PROJECTION);
  gluOrtho2D(0.0, g_screen->get_window_width(), 0.0, g_screen->get_window_height());
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  unsigned int size = g_screen->get_window_width()
    * g_screen->get_window_height() * sizeof(cl_float3);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // create OpenCL buffer from OpenGL vertex buffer object
  cl_vbo = cl::BufferGL(params.context, CL_MEM_WRITE_ONLY, vbo);
  cl_vbos.push_back(cl_vbo);
}

static void key_event(char key, bool down) {
}

static void mouse_motion_event(float xrel, float yrel, int x, int y) {
  // cam->update_view_angles(xrel, yrel);
}

static void mouse_button_event(int button, bool down, int x, int y) {
}

static void update(double dt, double t) {
  cpu_spheres[6].position.s[0] = -0.25f + cos((t * 10.f) / 5.f) / 8.f;
  cpu_spheres[6].position.s[1] = sin((t * 10.f) / 11.f) / 10.f;
  cpu_spheres[6].position.s[2] = -0.1f + cos((t * 10.f) / 7.f) / 6.f;
}

static void draw(double alpha) {
  // glViewport(0, 0, g_screen->get_window_width(), g_screen->get_window_height());

  glFinish();

  size_t local_work_size =
    params.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(params.device)
    , global_work_size = g_screen->get_window_width()
    * g_screen->get_window_height();
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1)
      * local_work_size;

  params.kernel.setArg(0, cl_spheres);
  params.kernel.setArg(1, num_spheres);
  params.kernel.setArg(2, cl_vbo);
  params.kernel.setArg(3, g_screen->get_window_width());
  params.kernel.setArg(4, g_screen->get_window_height());

  params.queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0
      , num_spheres * sizeof(Sphere), cpu_spheres);

  params.queue.enqueueAcquireGLObjects(&cl_vbos);
  params.queue.enqueueNDRangeKernel(params.kernel, cl::NullRange
      , global_work_size, local_work_size);
  params.queue.enqueueReleaseGLObjects(&cl_vbos);
  params.queue.finish();

  glClear(GL_COLOR_BUFFER_BIT);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(2, GL_FLOAT, 16, 0); // size (2, 3 or 4), type, stride, pointer
  glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)8); // size (3 or 4), type, stride, pointer

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, g_screen->get_window_width() * g_screen->get_window_height());
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
}

static void cleanup() {
}

int main() {
  g_screen->mainloop(load, key_event, mouse_motion_event, mouse_button_event
      , update, draw, cleanup);
}

