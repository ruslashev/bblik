#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glew.h>
#include <GL/glx.h>
#include <CL/cl.hpp>
#include <GL/glut.h>

// TODO
// cleanup()
// check for cl-gl interop

using namespace std;
using namespace cl;

//#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"

const int window_width = 1280;
const int window_height = 720;

// OpenGL vertex buffer object
GLuint vbo;

void render();

void initGL(int argc, char** argv){
  // init GLUT for OpenGL viewport
  glutInit(&argc, argv);
  // specify the display mode to be RGB and single buffering
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  // specify the initial window position
  glutInitWindowPosition(50, 50);
  // specify the initial window size
  glutInitWindowSize(window_width, window_height);
  // create the window and set title
  glutCreateWindow("Basic OpenCL path tracer");

  // register GLUT callback function to display graphics:
  glutDisplayFunc(render);

  // initialise OpenGL extensions
  glewInit();

  // initialise OpenGL
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glMatrixMode(GL_PROJECTION);
  gluOrtho2D(0.0, window_width, 0.0, window_height);
}

void createVBO(GLuint* vbo) {
  //create vertex buffer object
  glGenBuffers(1, vbo);
  glBindBuffer(GL_ARRAY_BUFFER, *vbo);

  //initialise VBO
  unsigned int size = window_width * window_height * sizeof(cl_float3);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawGL(){
  //clear all pixels, then render from the vbo
  glClear(GL_COLOR_BUFFER_BIT);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(2, GL_FLOAT, 16, 0); // size (2, 3 or 4), type, stride, pointer
  glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)8); // size (3 or 4), type, stride, pointer

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, window_width * window_height);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // flip backbuffer to screen
  glutSwapBuffers();
}

void Timer(int value) {
  glutPostRedisplay();
  glutTimerFunc(15, Timer, 0);
}

const int sphere_count = 9;

// OpenCL objects
Device device;
CommandQueue queue;
Kernel kernel;
Context context;
Program program;
Buffer cl_output;
Buffer cl_spheres;
BufferGL cl_vbo;
vector<Memory> cl_vbos;

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;
unsigned int framenumber = 0;

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
};

Sphere cpu_spheres[sphere_count];

void initOpenCL() {
  vector<Platform> platforms;
  Platform::get(&platforms);
  cout << "Available OpenCL platforms: " << endl;
  for (int i = 0; i < platforms.size(); i++)
    printf("  %d. %s\n", i + 1, platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
  Platform platform = platforms[0];

  vector<Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cout << "Available OpenCL devices on this platform: " << endl;
  for (int i = 0; i < devices.size(); i++)
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

  context = Context(device, properties);

  queue = CommandQueue(context, device);

  ifstream ifs("opencl_kernel.cl");
  if (!ifs) {
    printf("missing file: \"opencl_kernel.cl\"\n");
    cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
    exit(1);
  }
  string source { istreambuf_iterator<char>(ifs), istreambuf_iterator<char>() };

  const char *source_c_str = source.c_str();

  program = Program(context, source_c_str);

  cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
  if (result) {
    printf("Failed to compile OpenCL program (%d)\n", result);
    if (result == CL_BUILD_PROGRAM_FAILURE) {
      string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      printf("Build log:\n%s\n", build_log.c_str());
    }
    exit(1);
  }
}

void initScene(Sphere* cpu_spheres) {
#define float3(x, y, z) {{x, y, z}}  // macro to replace ugly initializer braces
  // left wall
  cpu_spheres[0].radius = 200.0f;
  cpu_spheres[0].position = float3(-200.6f, 0.0f, 0.0f);
  cpu_spheres[0].color = float3(0.75f, 0.25f, 0.25f);
  cpu_spheres[0].emission = float3(0.0f, 0.0f, 0.0f);

  // right wall
  cpu_spheres[1].radius = 200.0f;
  cpu_spheres[1].position = float3(200.6f, 0.0f, 0.0f);
  cpu_spheres[1].color = float3(0.25f, 0.25f, 0.75f);
  cpu_spheres[1].emission = float3(0.0f, 0.0f, 0.0f);

  // floor
  cpu_spheres[2].radius = 200.0f;
  cpu_spheres[2].position = float3(0.0f, -200.4f, 0.0f);
  cpu_spheres[2].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[2].emission = float3(0.0f, 0.0f, 0.0f);

  // ceiling
  cpu_spheres[3].radius = 200.0f;
  cpu_spheres[3].position = float3(0.0f, 200.4f, 0.0f);
  cpu_spheres[3].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[3].emission = float3(0.0f, 0.0f, 0.0f);

  // back wall
  cpu_spheres[4].radius = 200.0f;
  cpu_spheres[4].position = float3(0.0f, 0.0f, -200.4f);
  cpu_spheres[4].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[4].emission = float3(0.0f, 0.0f, 0.0f);

  // front wall
  cpu_spheres[5].radius = 200.0f;
  cpu_spheres[5].position = float3(0.0f, 0.0f, 202.0f);
  cpu_spheres[5].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[5].emission = float3(0.0f, 0.0f, 0.0f);

  // left sphere
  cpu_spheres[6].radius = 0.16f;
  cpu_spheres[6].position = float3(-0.25f, -0.24f, -0.1f);
  cpu_spheres[6].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[6].emission = float3(0.0f, 0.0f, 0.0f);

  // right sphere
  cpu_spheres[7].radius = 0.16f;
  cpu_spheres[7].position = float3(0.25f, -0.24f, 0.1f);
  cpu_spheres[7].color = float3(0.9f, 0.8f, 0.7f);
  cpu_spheres[7].emission = float3(0.0f, 0.0f, 0.0f);

  // lightsource
  cpu_spheres[8].radius = 1.0f;
  cpu_spheres[8].position = float3(0.0f, 1.36f, 0.0f);
  cpu_spheres[8].color = float3(0.0f, 0.0f, 0.0f);
  cpu_spheres[8].emission = float3(9.0f, 8.0f, 6.0f);
}

void initCLKernel(){

  // pick a rendermode
  unsigned int rendermode = 1;

  // Create a kernel (entry point in the OpenCL source program)
  kernel = Kernel(program, "render_kernel");

  // specify OpenCL kernel arguments
  //kernel.setArg(0, cl_output);
  kernel.setArg(0, cl_spheres);
  kernel.setArg(1, window_width);
  kernel.setArg(2, window_height);
  kernel.setArg(3, sphere_count);
  kernel.setArg(4, cl_vbo);
  kernel.setArg(5, framenumber);
}

void runKernel(){
  // every pixel in the image has its own thread or "work item",
  // so the total amount of work items equals the number of pixels
  std::size_t global_work_size = window_width * window_height;
  std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);;

  // Ensure the global work size is a multiple of local work size
  if (global_work_size % local_work_size != 0)
    global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

  //Make sure OpenGL is done using the VBOs
  glFinish();

  //this passes in the vector of VBO buffer objects
  queue.enqueueAcquireGLObjects(&cl_vbos);
  queue.finish();

  // launch the kernel
  queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size); // local_work_size
  queue.finish();

  //Release the VBOs so OpenGL can play with them
  queue.enqueueReleaseGLObjects(&cl_vbos);
  queue.finish();
}


// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
unsigned int WangHash(unsigned int a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}


void render(){

  framenumber++;

  cpu_spheres[6].position.s[1] += 0.01;

  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_count * sizeof(Sphere), cpu_spheres);

  kernel.setArg(0, cl_spheres);
  kernel.setArg(5, WangHash(framenumber));

  runKernel();

  drawGL();
}

void cleanUp(){
  //	delete cpu_output;
}

int main(int argc, char** argv){

  // initialise OpenGL (GLEW and GLUT window + callback functions)
  initGL(argc, argv);
  cout << "OpenGL initialized \n";

  // initialise OpenCL
  initOpenCL();

  // create vertex buffer object
  createVBO(&vbo);

  // call Timer():
  Timer(0);

  //make sure OpenGL is finished before we proceed
  glFinish();

  // initialise scene
  initScene(cpu_spheres);

  cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_count * sizeof(Sphere));
  queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_count * sizeof(Sphere), cpu_spheres);

  // create OpenCL buffer from OpenGL vertex buffer object
  cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
  cl_vbos.push_back(cl_vbo);

  // intitialise the kernel
  initCLKernel();

  // start rendering continuously
  glutMainLoop();

  // release memory
  cleanUp();

}
