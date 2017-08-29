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
  cl::size_t<3> dims;
  std::vector<cl::Memory> objs;
} params;
cl::Buffer cl_spheres;

struct render_params {
  shader_program *sp;
  GLuint vao;
  GLuint tex;
  int mat_loc, tex_loc;
} rparams;

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

  params.dims[0] = g_screen->get_window_width();
  params.dims[1] = g_screen->get_window_height();
  params.dims[2] = 1;

  cl_spheres = cl::Buffer(params.context, CL_MEM_READ_ONLY
      , num_spheres * sizeof(Sphere));

  // create opengl stuff
  glClearColor(0.2, 0.2, 0.2, 1.0);

  std::string vertex_shader_source = read_file_to_string("screen.vert")
    , fragment_shader_source = read_file_to_string("screen.frag");
  rparams.sp = new shader_program(vertex_shader_source, fragment_shader_source);
  rparams.mat_loc = rparams.sp->bind_uniform("matrix");
  rparams.tex_loc = rparams.sp->bind_uniform("tex");

  glGenTextures(1, &rparams.tex);
  glBindTexture(GL_TEXTURE_2D, rparams.tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // need to set GL_NEAREST
  // (not GL_NEAREST_MIPMAP_* which would cause CL_INVALID_GL_OBJECT later)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA /*GL_RGBA8*/
      , g_screen->get_window_width(), g_screen->get_window_height(), 0
      , GL_RGBA, GL_FLOAT, 0);

  array_buffer vbo;
  const std::vector<float> screen_vertices = {
    -1.f, -1.f,
     1.f, -1.f,
     1.f,  1.f,
    -1.f,  1.f
  };
  vbo.bind();
  vbo.upload(screen_vertices);
  array_buffer tbo;
  const std::vector<float> screen_texcords = {
    0., 0., // y flipped
    1., 0.,
    1., 1.,
    0., 1.
  };
  tbo.bind();
  tbo.upload(screen_texcords);
  element_array_buffer ebo;
  const std::vector<GLushort> elements = { 0, 1, 2, 0, 2, 3 };
  ebo.bind();
  ebo.upload(elements);
  // bind vao and attach vbo, tbo and ebo
  glGenVertexArrays(1,&rparams.vao);
  glBindVertexArray(rparams.vao);
  vbo.bind();
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(0);
  tbo.bind();
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glEnableVertexAttribArray(1);
  ebo.bind();
  glBindVertexArray(0);

  // create opencl texture reference using opengl texture
  cl_int err_code;
  params.tex = cl::ImageGL(params.context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0
      , rparams.tex, &err_code);
  assertf(err_code == CL_SUCCESS, "Failed to create OpenGL texture refrence "
      "(%d)", err_code);
  params.objs.push_back(params.tex);
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
  glViewport(0, 0, g_screen->get_window_width(), g_screen->get_window_height());

  glFinish();

  params.queue.enqueueWriteBuffer(cl_spheres, CL_FALSE, 0
      , num_spheres * sizeof(Sphere), cpu_spheres);

  params.queue.enqueueAcquireGLObjects(&params.objs);

  params.kernel.setArg(0, cl_spheres);
  params.kernel.setArg(1, num_spheres);
  params.kernel.setArg(2, params.tex);
  params.kernel.setArg(3, g_screen->get_window_width());
  params.kernel.setArg(4, g_screen->get_window_height());

  size_t local_work_size = params.kernel.getWorkGroupInfo<
    CL_KERNEL_WORK_GROUP_SIZE>(params.device)
    , global_work_size = g_screen->get_window_width()
    * g_screen->get_window_height();
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1)
      * local_work_size;

  params.queue.enqueueNDRangeKernel(params.kernel, cl::NullRange
      , global_work_size, local_work_size);

  params.queue.enqueueReleaseGLObjects(&params.objs);
  params.queue.finish();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  rparams.sp->use_this_prog();
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(rparams.tex_loc, 0);
  glBindTexture(GL_TEXTURE_2D, rparams.tex);
  glUniformMatrix4fv(rparams.mat_loc, 1, GL_FALSE, proj_matrix);
  glBindVertexArray(rparams.vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
  glBindVertexArray(0);
}

static void cleanup() {
}

int main() {
  g_screen->mainloop(load, key_event, mouse_motion_event, mouse_button_event
      , update, draw, cleanup);
}

