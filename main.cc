#include "screen.hh"
#include "ogl.hh"
#include "ocl_helpers.hh"
#include "ogl_helpers.hh"
#include <GL/glx.h>

static const uint NUM_JSETS = 9;

static const float matrix[16] =
{
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 1.0f
};

static const float vertices[12] =
{
  -1.0f,-1.0f, 0.0,
  1.0f,-1.0f, 0.0,
  1.0f, 1.0f, 0.0,
  -1.0f, 1.0f, 0.0
};

static const float texcords[8] =
{
  0.0, 1.0,
  1.0, 1.0,
  1.0, 0.0,
  0.0, 0.0
};

static const uint indices[6] = {0,1,2,0,2,3};

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

typedef struct {
  Device d;
  CommandQueue q;
  Program p;
  Kernel k;
  ImageGL tex;
  cl::size_t<3> dims;
} process_params;

typedef struct {
  GLuint prg;
  GLuint vao;
  GLuint tex;
} render_params;

process_params params;
render_params rparams;

screen *g_screen = new screen("bblik", 800, 600);

void load() {
  Platform lPlatform = getPlatform();
  // Select the default platform and create a context using this platform and the GPU
  cl_context_properties cps[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)lPlatform(),
    0
  };
  std::vector<Device> devices;
  lPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  // Get a list of devices on this platform
  for (unsigned d=0; d<devices.size(); ++d) {
    if (checkExtnAvailability(devices[d],CL_GL_SHARING_EXT)) {
      params.d = devices[d];
      break;
    }
  }
  Context context(params.d, cps);
  // Create a command queue and use the first device
  params.q = CommandQueue(context, params.d);
  cl_int errCode;
  params.p = getProgram(context, "fractal.cl",errCode);

  std::ostringstream options;
  options << "";

  params.p.build(std::vector<Device>(1, params.d), options.str().c_str());
  params.k = Kernel(params.p, "fractal");
  // create opengl stuff
  rparams.prg = initShaders("fractal.vert", "fractal.frag");

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

  GLuint vbo  = createBuffer(12,vertices,GL_STATIC_DRAW);
  GLuint tbo  = createBuffer(8,texcords,GL_STATIC_DRAW);
  GLuint ibo;
  glGenBuffers(1,&ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(uint)*6,indices,GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
  // bind vao
  glGenVertexArrays(1,&rparams.vao);
  glBindVertexArray(rparams.vao);
  // attach vbo
  glBindBuffer(GL_ARRAY_BUFFER,vbo);
  glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,NULL);
  glEnableVertexAttribArray(0);
  // attach tbo
  glBindBuffer(GL_ARRAY_BUFFER,tbo);
  glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,0,NULL);
  glEnableVertexAttribArray(1);
  // attach ibo
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo);
  glBindVertexArray(0);
  // create opengl texture reference using opengl texture
  params.tex = ImageGL(context,CL_MEM_READ_WRITE,GL_TEXTURE_2D,0,rparams.tex,&errCode);
  if (errCode!=CL_SUCCESS) {
    std::cout<<"Failed to create OpenGL texture refrence: "<<errCode<<std::endl;
  }
  params.dims[0] = g_screen->get_window_width();
  params.dims[1] = g_screen->get_window_height();
  params.dims[2] = 1;
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

  cl::Event ev;
  glFinish();

  std::vector<Memory> objs;
  objs.clear();
  objs.push_back(params.tex);
  // flush opengl commands and wait for object acquisition
  cl_int res = params.q.enqueueAcquireGLObjects(&objs,NULL,&ev);
  ev.wait();
  if (res!=CL_SUCCESS) {
    std::cout<<"Failed acquiring GL object: "<<res<<std::endl;
    exit(248);
  }
  NDRange local(16, 16);
  NDRange global( local[0] * divup(params.dims[0], local[0]),
      local[1] * divup(params.dims[1], local[1]));
  // set kernel arguments
  params.k.setArg(0, params.tex);
  params.k.setArg(1, (int)params.dims[0]);
  params.k.setArg(2, (int)params.dims[1]);
  params.k.setArg(3, 1.0f);
  params.k.setArg(4, 1.0f);
  params.k.setArg(5, 0.0f);
  params.k.setArg(6, 0.0f);
  params.k.setArg(7, CJULIA[2*0+0]);
  params.k.setArg(8, CJULIA[2*0+1]);
  params.q.enqueueNDRangeKernel(params.k,cl::NullRange, global, local);
  // release opengl object
  res = params.q.enqueueReleaseGLObjects(&objs);
  ev.wait();
  if (res!=CL_SUCCESS) {
    std::cout<<"Failed releasing GL object: "<<res<<std::endl;
    exit(247);
  }
  params.q.finish();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glClearColor(0.2,0.2,0.2,0.0);
  glEnable(GL_DEPTH_TEST);
  // bind shader
  glUseProgram(rparams.prg);
  // get uniform locations
  int mat_loc = glGetUniformLocation(rparams.prg,"matrix");
  int tex_loc = glGetUniformLocation(rparams.prg,"tex");
  // bind texture
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(tex_loc,0);
  glBindTexture(GL_TEXTURE_2D,rparams.tex);
  glGenerateMipmap(GL_TEXTURE_2D);
  // set project matrix
  glUniformMatrix4fv(mat_loc,1,GL_FALSE,matrix);
  // now render stuff
  glBindVertexArray(rparams.vao);
  glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);
  glBindVertexArray(0);
}

static void cleanup() {
}

int main() {
  g_screen->mainloop(load, key_event, mouse_motion_event, mouse_button_event
      , update, draw, cleanup);
}

