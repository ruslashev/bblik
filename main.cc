#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <GL/glew.h>
#include <GL/glx.h>
#include <CL/cl.hpp>
#include <GL/glut.h>

// padding with dummy variables are required for memory alignment
// float3 is considered as float4 by OpenCL
// alignment can also be enforced by using __attribute__ ((aligned (16)));
// see https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/attributes-variables.html
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

const int window_width = 1280;
const int window_height = 720;

GLuint vbo;

cl::Device device;
cl::CommandQueue queue;
cl::Kernel kernel;
cl::Context context;
cl::Program program;
cl::Buffer cl_output;
cl::Buffer cl_spheres;
cl::BufferGL cl_vbo;
std::vector<cl::Memory> cl_vbos;

unsigned int frame_number = 0;

#define _float3(x, y, z) {{x, y, z}}  // macro to replace ugly initializer braces

#if 0
Sphere cpu_spheres[] = {
  Sphere(1e5,  _float3( 1e5 + 1, 40.8, 81.6),   _float3(.75,.25,.25),       _float3( 0,  0,  0)),
  Sphere(1e5,  _float3(-1e5 + 99, 40.8, 81.6),  _float3(.25,.25,.75),       _float3( 0,  0,  0)),
  Sphere(1e5,  _float3(50, 40.8, 1e5),          _float3(.25,.75,.25),       _float3( 0,  0,  0)),
  Sphere(1e5,  _float3(50, 1e5, 81.6),          _float3(.75,.75,.75),       _float3( 0,  0,  0)),
  Sphere(1e5,  _float3(50,-1e5 + 81.6, 81.6),   _float3(.75,.75,.75),       _float3( 0,  0,  0)),
  Sphere(16.5, _float3(27, 16.5, 47),           _float3(.999, .999, .999),  _float3( 0,  0,  0)),
  Sphere(16.5, _float3(73, 16.5, 78),           _float3(.999, .999, .999),  _float3( 0,  0,  0)),
  Sphere(600,  _float3(50, 681.6 - .27, 81.6),  _float3(0, 0, 0),           _float3(12, 12, 12)),
};
#endif
Sphere cpu_spheres[] = {
  Sphere(200.f, _float3(-200.6f, 0.0f, 0.0f),   _float3(0.75f, 0.25f, 0.25f), _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(200.f, _float3(200.6f, 0.0f, 0.0f),    _float3(0.25f, 0.25f, 0.75f), _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(200.f, _float3(0.0f, -200.4f, 0.0f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(200.f, _float3(0.0f, 200.4f, 0.0f),    _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(200.f, _float3(0.0f, 0.0f, -200.4f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(200.f, _float3(0.0f, 0.0f, 202.0f),    _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(0.16f, _float3(-0.25f, -0.24f, -0.1f), _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(0.16f, _float3(0.25f, -0.24f, 0.1f),   _float3(0.9f, 0.8f, 0.7f),    _float3(0.0f, 0.0f, 0.0f) ),
  Sphere(  1.f, _float3(0.0f, 1.36f, 0.0f),     _float3(0.0f, 0.0f, 0.0f),    _float3(9.0f, 8.0f, 6.0f) )
};
const int num_spheres = sizeof(cpu_spheres) / sizeof(Sphere);

void render();

void initGL(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(window_width, window_height);
  glutCreateWindow("bblik");

  glutDisplayFunc(render);

  glewInit();

  glClearColor(0.0, 0.0, 0.0, 1.0);

  // glPushMatrix for normal opengl drawing?
  glMatrixMode(GL_PROJECTION);
  gluOrtho2D(0.0, window_width, 0.0, window_height);
}

void createVBO(GLuint* vbo) {
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);
  unsigned int size = window_width * window_height * sizeof(cl_float3);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawGL() {
  glClear(GL_COLOR_BUFFER_BIT);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(2, GL_FLOAT, 16, 0);
  glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)8);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, window_width * window_height);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  glutSwapBuffers();
}

void Timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(15, Timer, 0);
}
void initOpenCL() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  printf("Available OpenCL platforms: \n");
  for (size_t i = 0; i < platforms.size(); i++)
    printf("  %d. %s\n", i + 1, platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
  cl::Platform platform = platforms[0];

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  printf("Available OpenCL devices on this platform: \n");
  for (size_t i = 0; i < devices.size(); i++)
    printf("  %d. %s (max compute units %d, max work group size %d)\n"
        , i + 1
        , devices[i].getInfo<CL_DEVICE_NAME>().c_str()
        , devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
        , devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

  device = devices[0];

  cl_context_properties properties[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
    0
  };

  context = cl::Context(device, properties);

  queue = cl::CommandQueue(context, device);

  std::ifstream ifs("opencl_kernel.cl");
  if (!ifs) {
    printf("missing file: \"opencl_kernel.cl\"\n");
    exit(1);
  }
  std::string source { std::istreambuf_iterator<char>(ifs)
    , std::istreambuf_iterator<char>() };

  const char *source_c_str = source.c_str();

  program = cl::Program(context, source_c_str);

  cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
  if (result) {
    printf("Failed to compile OpenCL program (%d)\n", result);
    if (result == CL_BUILD_PROGRAM_FAILURE) {
      std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      printf("Build log:\n%s\n", build_log.c_str());
    }
    exit(1);
  }
}

void initCLKernel() {
  kernel = cl::Kernel(program, "render_kernel");

  kernel.setArg(0, cl_spheres);
  kernel.setArg(1, num_spheres);
  kernel.setArg(2, window_width);
  kernel.setArg(3, window_height);
  kernel.setArg(4, cl_vbo);
}

void runKernel() {
  // every pixel in the image has its own thread or "work item",
  // so the total amount of work items equals the number of pixels
  std::size_t global_work_size = window_width * window_height;
  std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  // Make sure OpenGL is done using the VBOs
  glFinish();

  // this passes in the vector of VBO buffer objects
  queue.enqueueAcquireGLObjects(&cl_vbos);
  queue.finish();

  // launch the kernel
  queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size); // local_work_size
  queue.finish();

  // Release the VBOs so OpenGL can play with them
  queue.enqueueReleaseGLObjects(&cl_vbos);
  queue.finish();
}

void render() {
  const auto time_render_begin = std::chrono::high_resolution_clock::now();

  ++frame_number;

  cpu_spheres[6].position.s[1] = sin((float)frame_number / 11.f) / 10.f;
  cpu_spheres[6].position.s[0] = -0.25f + cos((float)frame_number / 7.f) / 10.f;

  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, num_spheres * sizeof(Sphere), cpu_spheres);

  kernel.setArg(0, cl_spheres);

  runKernel();

  drawGL();

  const auto time_render_end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> render_duration
    = time_render_end - time_render_begin;
  const std::string title_string = "bblik " + std::to_string(render_duration.count())
    + " us/f, " + std::to_string(1. / (render_duration.count() / 1000.f)) + " f/s";
  glutSetWindowTitle(title_string.c_str());
}

int main(int argc, char** argv) {
  initGL(argc, argv);

  initOpenCL();

  createVBO(&vbo);

  Timer(0);

  // make sure OpenGL is finished before we proceed
  glFinish();

  cl_spheres = cl::Buffer(context, CL_MEM_READ_ONLY, num_spheres * sizeof(Sphere));
  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, num_spheres * sizeof(Sphere), cpu_spheres);

  // create OpenCL buffer from OpenGL vertex buffer object
  cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
  cl_vbos.push_back(cl_vbo);

  initCLKernel();

  // start rendering continuously
  glutMainLoop();
}

