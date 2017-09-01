#include "screen.hh"
#include "utils.hh"
#include <chrono>

screen::screen(const std::string &n_title, int n_window_width
    , int n_window_height)
  : _title(n_title)
  , _pre_lock_mouse_x(n_window_width / 2)
  , _pre_lock_mouse_y(n_window_height / 2)
  , _window_width(n_window_width)
  , _window_height(n_window_height)
  , _frame_idx(0)
  , running(true) {
  assertf(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) == 0
      , "failed to init sdl: %s", SDL_GetError());

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

  _window = SDL_CreateWindow(_title.c_str(), SDL_WINDOWPOS_CENTERED
      , SDL_WINDOWPOS_CENTERED, _window_width, _window_height, SDL_WINDOW_OPENGL);

  _gl_context = SDL_GL_CreateContext(_window);

  if (SDL_GL_SetSwapInterval(0) == -1)
    warning("failed to set vsync: %s", SDL_GetError());

  GLenum err = glewInit();
  assertf(err == GLEW_OK, "failed to initialze glew: %s",
      glewGetErrorString(err));

  assertf(GLEW_VERSION_2_1, "your graphic card does not support OpenGL 2.1");
}

screen::~screen() {
  SDL_GL_DeleteContext(_gl_context);
  SDL_DestroyWindow(_window);
  SDL_Quit();
}

static char sdlkey_to_char(const SDL_Keycode &kc) {
  switch (kc) {
    case SDLK_a: return 'a';
    case SDLK_c: return 'c';
    case SDLK_d: return 'd';
    case SDLK_f: return 'f';
    case SDLK_q: return 'q';
    case SDLK_s: return 's';
    case SDLK_w: return 'w';
    case SDLK_x: return 'x';
    case SDLK_z: return 'z';
    default:     return -1;
  }
}

void screen::mainloop(void (*load_cb)(void)
    , void (*key_event_cb)(char, bool)
    , void (*mouse_motion_event_cb)(float, float, int, int)
    , void (*mouse_button_event_cb)(int, bool, int, int)
    , void (*update_cb)(double, double)
    , void (*draw_cb)(double)
    , void (*cleanup_cb)(void)) {
  load_cb();

  const int ticks_per_second = 30, max_update_ticks = 15;
  // everywhere all time is measured in seconds unless otherwise stated
  double t = 0, dt = 1. / ticks_per_second;
  double current_time = get_time_in_seconds(), accumulator = 0;

  int draw_count = 0;

  while (running) {
    double real_time = get_time_in_seconds()
      , elapsed = real_time - current_time;
    elapsed = std::min(elapsed, max_update_ticks * dt);
    current_time = real_time;
    accumulator += elapsed;

    while (accumulator >= dt) {
      { // events
        SDL_Event sdl_event;
        while (SDL_PollEvent(&sdl_event) != 0)
          if (sdl_event.type == SDL_QUIT)
            running = false;
          else if ((sdl_event.type == SDL_KEYDOWN || sdl_event.type == SDL_KEYUP)
              && sdl_event.key.repeat == 0) {
            const char key_info = sdlkey_to_char(sdl_event.key.keysym.sym);
            if (key_info != -1)
              key_event_cb(key_info, sdl_event.type == SDL_KEYDOWN);
          } else if (sdl_event.type == SDL_MOUSEMOTION)
            mouse_motion_event_cb(static_cast<float>(sdl_event.motion.xrel)
                , static_cast<float>(sdl_event.motion.yrel), sdl_event.motion.x
                , sdl_event.motion.y);
          else if (sdl_event.type == SDL_MOUSEBUTTONDOWN
              || sdl_event.type == SDL_MOUSEBUTTONUP) {
            int button;
            switch (sdl_event.button.button) {
              case SDL_BUTTON_LEFT:   button = 1; break;
              case SDL_BUTTON_MIDDLE: button = 2; break;
              case SDL_BUTTON_RIGHT:  button = 3; break;
              case SDL_BUTTON_X1:     button = 4; break;
              case SDL_BUTTON_X2:     button = 5; break;
              default:                button = -1;
            }
            mouse_button_event_cb(button, sdl_event.type == SDL_MOUSEBUTTONDOWN
                , sdl_event.motion.x, sdl_event.motion.y);
          }
      }

      update_cb(dt, t);

      t += dt;
      accumulator -= dt;
    }

    auto draw_begin_w = std::chrono::high_resolution_clock::now();
    std::clock_t draw_begin_c = std::clock();
    draw_cb(accumulator / dt);
    auto draw_end_w = std::chrono::high_resolution_clock::now();
    std::clock_t draw_end_c = std::clock();
    std::chrono::duration<double, std::milli> draw_duration_w = draw_end_w
      - draw_begin_w;
    float draw_duration_c = ((float)(draw_end_c - draw_begin_c) / CLOCKS_PER_SEC)
      * 1000.f;

    SDL_GL_SwapWindow(_window);

    ++_frame_idx;

    { // fps counter
      draw_count++;
      if (1 || draw_count == 700) {
        draw_count = 0;
        double now = get_time_in_seconds()
          , seconds_per_frame = now - real_time
          , fps = 1. / seconds_per_frame
          , fpsavg = (double)_frame_idx / now
          , mspf = seconds_per_frame * 1000.;
        char title[256];
        snprintf(title, 256, "%s | %7.2f ms/f, %7.2f f/s, %7.2f f/s avg"
            ", %.3f ms/d (wall) %.3f ms/d (cpu)", _title.c_str(), mspf, fps
            , fpsavg, draw_duration_w.count(), draw_duration_c);
        SDL_SetWindowTitle(_window, title);
      }
    }
  }

  cleanup_cb();
}

void screen::lock_mouse() {
  SDL_GetMouseState(&_pre_lock_mouse_x, &_pre_lock_mouse_y);
  SDL_SetRelativeMouseMode(SDL_TRUE);
}

void screen::unlock_mouse() {
  SDL_SetRelativeMouseMode(SDL_FALSE);
  SDL_WarpMouseInWindow(_window, _pre_lock_mouse_x, _pre_lock_mouse_y);
}

inline double screen::get_time_in_seconds() {
  return SDL_GetTicks() / 1000.;
}

int screen::get_window_width() {
  return _window_width;
}

int screen::get_window_height() {
  return _window_height;
}

unsigned long long int screen::get_frame_idx() {
  return _frame_idx;
}

