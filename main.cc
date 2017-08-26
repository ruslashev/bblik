#include "screen.hh"
#include "ogl.hh"
#include <GL/glx.h>
#include <CL/cl.hpp>

static const uint NUM_JSETS = 9;

static const float matrix[16] =
{
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 1.0f
};

static const float CJULIA[] = {
  -0.700f, 0.270f,
  -0.618f, 0.000f,
  -0.400f, 0.600f,
  0.285f, 0.000f,
  0.285f, 0.010f,
  0.450f, 0.143f,
  -0.702f,-0.384f,
  -0.835f,-0.232f,
  -0.800f, 0.156f,
  0.279f, 0.000f
};

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

struct render_params {
  shader_program *sp;
  GLuint vao;
  GLuint tex;
  int mat_loc, tex_loc;
} rparams;

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

  std::string source = read_file_to_string("fractal.cl");
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

  params.kernel = cl::Kernel(params.program, "fractal");

  params.dims[0] = g_screen->get_window_width();
  params.dims[1] = g_screen->get_window_height();
  params.dims[2] = 1;

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
  const std::vector<float> vertices = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f,
    -1.0f,  1.0f
  };
  vbo.bind();
  vbo.upload(vertices);
  array_buffer tbo;
  const std::vector<float> texcords = {
    0.0, 1.0,
    1.0, 1.0,
    1.0, 0.0,
    0.0, 0.0
  };
  tbo.bind();
  tbo.upload(texcords);
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
  params.tex = cl::ImageGL(params.context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0
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
}

unsigned divup(unsigned a, unsigned b) {
  return (a+b-1)/b;
}

static void draw(double alpha) {
  glViewport(0, 0, g_screen->get_window_width(), g_screen->get_window_height());

  glFinish();

  params.queue.enqueueAcquireGLObjects(&params.objs);

  size_t local_work_size = params.kernel.getWorkGroupInfo<
    CL_KERNEL_WORK_GROUP_SIZE>(params.device)
    , global_work_size = g_screen->get_window_width()
    * g_screen->get_window_height();
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1)
      * local_work_size;

  params.kernel.setArg(0, params.tex);
  params.kernel.setArg(1, (int)params.dims[0]);
  params.kernel.setArg(2, (int)params.dims[1]);
  params.kernel.setArg(3, 1.0f);
  params.kernel.setArg(4, 1.0f);
  params.kernel.setArg(5, 0.0f);
  params.kernel.setArg(6, 0.0f);
  params.kernel.setArg(7, CJULIA[2*0+0]);
  params.kernel.setArg(8, CJULIA[2*0+1]);
  params.queue.enqueueNDRangeKernel(params.kernel, cl::NullRange
      , global_work_size, local_work_size);

  params.queue.enqueueReleaseGLObjects(&params.objs);
  params.queue.finish();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  rparams.sp->use_this_prog();
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(rparams.tex_loc, 0);
  glBindTexture(GL_TEXTURE_2D, rparams.tex);
  glUniformMatrix4fv(rparams.mat_loc, 1, GL_FALSE, matrix);
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

