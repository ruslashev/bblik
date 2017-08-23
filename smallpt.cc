#include <algorithm> // std::max
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <glm/glm.hpp>

typedef glm::tvec3<double> vec3_t;

struct ray_t { vec3_t o, d; ray_t(vec3_t o_, vec3_t d_) : o(o_), d(d_) {} };

enum refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct sphere_t {
  double radius;
  vec3_t pos, emission, color;
  refl_t refl;
  sphere_t(double radius_, vec3_t pos_, vec3_t emission_, vec3_t color_, refl_t refl_)
    : radius(radius_), pos(pos_), emission(emission_), color(color_) , refl(refl_) {}
  double intersect(const ray_t &r) const { // returns distance, 0 if nohit
    vec3_t op = pos - r.o; // Solve t^2*d.d + 2*t*(o-pos).d + (o-pos).(o-pos)-R^2 = 0
    double t, eps = 1e-4, b = glm::dot(op, r.d)
      , det = b * b - glm::dot(op, op) + radius * radius;
    if (det < 0)
      return 0;
    else
      det = sqrt(det);
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

sphere_t spheres[] = {
  sphere_t(1e5,  vec3_t( 1e5 + 1, 40.8, 81.6),  vec3_t(), vec3_t(.75,.25,.25),  DIFF),//Left
  sphere_t(1e5,  vec3_t(-1e5 + 99, 40.8, 81.6), vec3_t(), vec3_t(.25,.25,.75),  DIFF),//Rght
  sphere_t(1e5,  vec3_t(50, 40.8, 1e5),         vec3_t(), vec3_t(.25,.75,.25),  DIFF),//Back
//sphere_t(1e5,  vec3_t(50, 40.8,-1e5 + 170),   vec3_t(), vec3_t(),             DIFF),//Frnt
  sphere_t(1e5,  vec3_t(50, 1e5, 81.6),         vec3_t(), vec3_t(.75,.75,.75),  DIFF),//Botm
  sphere_t(1e5,  vec3_t(50,-1e5 + 81.6, 81.6),  vec3_t(), vec3_t(.75,.75,.75),  DIFF),//Top
  sphere_t(16.5, vec3_t(27, 16.5, 47),          vec3_t(), vec3_t(1, 1, 1) * .999, SPEC),//Mirr
  sphere_t(16.5, vec3_t(73, 16.5, 78),          vec3_t(), vec3_t(1, 1, 1) * .999, REFR),//Glas
  sphere_t(600,  vec3_t(50, 681.6 - .27, 81.6), vec3_t(12, 12, 12),  vec3_t(),  DIFF) //Lite
};
const int num_spheres = sizeof(spheres) / sizeof(sphere_t);
const double infinity = 1e20;

double clamp(double x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

double linear_to_srgb(double x) {
  if (x < 0.0031308)
    x *= 12.92;
  else
    x = 1.055 * pow(x, 1.0 / 2.4) - 0.055;
  return x;
}

void vec_to_rgb(const vec3_t &input, int &r, int &g, int &b) {
  r = linear_to_srgb(input.x) * 255.0 + 0.5;
  g = linear_to_srgb(input.y) * 255.0 + 0.5;
  b = linear_to_srgb(input.z) * 255.0 + 0.5;
}

bool intersect(const ray_t &r, double &t, int &id) {
  double d;
  t = infinity;
  for (int i=0; i < num_spheres; ++i) {
    double d = spheres[i].intersect(r);
    if (d != 0. && d < t) {
      t = d;
      id = i;
    }
  }
  return t < infinity;
}

double randf() {
  // return (double)rand() / (double)RAND_MAX;
  return drand48();
}

vec3_t radiance(const ray_t &r, int depth) {
  double t; // distance to intersection
  int id = 0; // id of intersected object
  if (!intersect(r, t, id))
    return vec3_t(); // return black on miss

  const sphere_t &obj = spheres[id]; // the hit object
  vec3_t x = r.o + r.d * t,
         n = glm::normalize(x - obj.pos),
         nl = glm::dot(n, r.d) < 0 ? n : -n, f = obj.color;
  double p = std::max(f.x, std::max(f.y, f.z));

  // R.R.
  if (++depth > 5)
    if (randf() < p)
      f = f * (1. / p);
    else
      return obj.emission;

  if (obj.refl == DIFF) { // Ideal DIFFUSE reflection
    double r1 = 2 * M_PI * randf(), r2 = randf(), r2s = sqrt(r2);
    vec3_t w = nl
      , u = glm::normalize(glm::cross(fabs(w.x) > 0.1 ? vec3_t(0, 1, 0) : vec3_t(1, 0, 0), w))
      , v = glm::cross(w,u);
    vec3_t d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1. - r2));
    return obj.emission + f * radiance(ray_t(x, d), depth);
  } else if (obj.refl == SPEC) // Ideal SPECULAR reflection
    return obj.emission + f * radiance(ray_t(x, r.d - n * 2. * glm::dot(n, r.d)), depth);
  else if (obj.refl == REFR) { // Ideal dielectric REFRACTION
    ray_t refl_ray(x, r.d - n * 2. * glm::dot(n, r.d));
    bool into = glm::dot(n, nl) > 0; // Ray from outside going in?
    double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = glm::dot(r.d, nl), cos2t;
    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) // Total internal reflection
      return obj.emission + f * radiance(refl_ray, depth);
    vec3_t tdir = glm::normalize(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
    double a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? - ddn : glm::dot(tdir, n));
    double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re
      , P = 0.25 + 0.5 * Re, RP = Re / P, TP = Tr / (1 - P);
    return obj.emission + f * (depth > 2 ? (randf() < P ? // Russian roulette
          radiance(refl_ray, depth) * RP : radiance(ray_t(x, tdir), depth) * TP) :
        radiance(refl_ray, depth) * Re + radiance(ray_t(x, tdir), depth) * Tr);
  }
}

int main(int argc, char *argv[]) {
  int w = 1024, h = 768, samples = argc == 2 ? atoi(argv[1]) : 1;
  ray_t cam(vec3_t(50, 52, 295.6), glm::normalize(vec3_t(0, -0.042612, -1))); // cam pos, dir
  vec3_t cx = vec3_t(w * 0.5135 / h, 0, 0)
    , cy = glm::normalize(glm::cross(cx, cam.d)) * 0.5135, r
    , *c = new vec3_t[w * h];
#pragma omp parallel for schedule(dynamic, 1) private(r) // OpenMP
  for (int y = 0; y < h; y++) { // Loop over image rows
    fprintf(stderr,"\rRendering (%d spp) %5.2f%%", samples, 100. * (y + 1) / h);
    for (int x = 0; x < w; x++) { // Loop cols
      int i = (h - (y + 1)) * w + x; // y-inverted coordinate x,y in c array
      r.x = r.y = r.z = 0;
      for (int s = 0; s < samples; s++) {
        double r1 = 2. * randf(), dx = r1 < 1. ? sqrt(r1) - 1. : 1. - sqrt(2. - r1);
        double r2 = 2. * randf(), dy = r2 < 1. ? sqrt(r2) - 1. : 1. - sqrt(2. - r2);
        vec3_t d = cx * (((1. + dx) / 2. + x) / w - 0.5)
          + cy * (((1. + dy) / 2. + y) / h - 0.5) + cam.d;
        r += radiance(ray_t(cam.o + d, glm::normalize(d)), 0) * (1. / samples);
      } // Camera rays are pushed ^^^^^ forward to start in interior
      c[i] = vec3_t(clamp(r.x), clamp(r.y), clamp(r.z));
    }
  }
  puts("");
  FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i = 0; i < w * h; i++) {
    int r, g, b;
    vec_to_rgb(c[i], r, g, b);
    fprintf(f,"%d %d %d ", r, g, b);
  }
}

