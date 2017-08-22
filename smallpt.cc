#include <cmath>
#include <cstdlib>
#include <cstdio>

struct vec3_t {
  double x, y, z;
  vec3_t(double x_ = 0, double y_ = 0, double z_ = 0){ x = x_; y = y_; z = z_; }
  vec3_t operator+(const vec3_t &b) const { return vec3_t(x + b.x, y + b.y, z + b.z); }
  vec3_t operator-(const vec3_t &b) const { return vec3_t(x - b.x, y - b.y, z - b.z); }
  vec3_t operator*(double b) const { return vec3_t(x * b, y * b, z * b); }
  vec3_t mult(const vec3_t &b) const { return vec3_t(x * b.x, y * b.y, z * b.z); }
  vec3_t& norm(){ return *this =  *this * (1/sqrt(x * x + y * y + z * z)); }
  double dot(const vec3_t &b) const { return x * b.x + y * b.y + z * b.z; } // cross:
  vec3_t operator%(vec3_t&b){return vec3_t(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);}
};

struct ray_t { vec3_t o, d; ray_t(vec3_t o_, vec3_t d_) : o(o_), d(d_) {} };

enum refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct sphere_t {
  double rad;       // radius
  vec3_t p, e, c;      // position, emission, color
  refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
  sphere_t(double rad_, vec3_t p_, vec3_t e_, vec3_t c_, refl_t refl_):
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
  double intersect(const ray_t &r) const { // returns distance, 0 if nohit
    vec3_t op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
    if (det<0) return 0; else det = sqrt(det);
    return (t = b - det)>eps ? t : ((t = b + det)>eps ? t : 0);
  }
};

sphere_t spheres[] = {//Scene: radius, position, emission, color, material
  sphere_t(1e5, vec3_t( 1e5 + 1, 40.8, 81.6), vec3_t(), vec3_t(.75,.25,.25), DIFF),//Left
  sphere_t(1e5, vec3_t(-1e5 + 99, 40.8, 81.6), vec3_t(), vec3_t(.25,.25,.75), DIFF),//Rght
  sphere_t(1e5, vec3_t(50, 40.8, 1e5),     vec3_t(), vec3_t(.75,.75,.75), DIFF),//Back
  sphere_t(1e5, vec3_t(50, 40.8,-1e5 + 170), vec3_t(), vec3_t(),           DIFF),//Frnt
  sphere_t(1e5, vec3_t(50, 1e5, 81.6),    vec3_t(), vec3_t(.75,.75,.75), DIFF),//Botm
  sphere_t(1e5, vec3_t(50,-1e5 + 81.6, 81.6), vec3_t(), vec3_t(.75,.75,.75), DIFF),//Top
  sphere_t(16.5, vec3_t(27, 16.5, 47),       vec3_t(), vec3_t(1, 1, 1)*.999, SPEC),//Mirr
  sphere_t(16.5, vec3_t(73, 16.5, 78),       vec3_t(), vec3_t(1, 1, 1)*.999, REFR),//Glas
  sphere_t(600, vec3_t(50, 681.6 - .27, 81.6), vec3_t(12, 12, 12),  vec3_t(), DIFF) //Lite
};
const int num_spheres = sizeof(spheres) / sizeof(sphere_t);
const double infinity = 1e20;

double clamp(double x) {
  return x<0 ? 0 : x>1 ? 1 : x;
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
  for (int i=0; i < num_spheres; ++i)
    if ((d = spheres[i].intersect(r)) && d < t) {
      t = d;
      id = i;
    }
  return t < infinity;
}

vec3_t radiance(const ray_t &r, int depth, unsigned short *Xi) {
  double t;                               // distance to intersection
  int id = 0;                               // id of intersected object
  if (!intersect(r, t, id)) return vec3_t(); // if miss, return black
  const sphere_t &obj = spheres[id];        // the hit object
  vec3_t x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d)<0?n:n * -1, f = obj.c;
  double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
  if (++depth>5) if (erand48(Xi)<p) f = f * (1/p); else return obj.e; //R.R.
  if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
    double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
    vec3_t w = nl, u = ((fabs(w.x)>.1?vec3_t(0, 1):vec3_t(1))%w).norm(), v = w%u;
    vec3_t d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
    return obj.e + f.mult(radiance(ray_t(x, d), depth, Xi));
  } else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
    return obj.e + f.mult(radiance(ray_t(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
  ray_t reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
  bool into = n.dot(nl)>0;                // ray_t from outside going in?
  double nc = 1, nt = 1.5, nnt = into?nc/nt:nt/nc, ddn = r.d.dot(nl), cos2t;
  if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn))<0)    // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, Xi));
  vec3_t tdir = (r.d * nnt  -  n * ((into?1: -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  double a = nt - nc, b = nt + nc, R0 = a * a/(b * b), c = 1 - (into? - ddn:tdir.dot(n));
  double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re/P, TP = Tr/(1 - P);
  return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette
    radiance(reflRay, depth, Xi) * RP:radiance(ray_t(x, tdir), depth, Xi) * TP) :
    radiance(reflRay, depth, Xi) * Re + radiance(ray_t(x, tdir), depth, Xi) * Tr);
}

int main(int argc, char *argv[]) {
  int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1])/4 : 1; // # samples
  ray_t cam(vec3_t(50, 52, 295.6), vec3_t(0, -0.042612, -1).norm()); // cam pos, dir
  vec3_t cx = vec3_t(w * .5135/h), cy = (cx%cam.d).norm() * .5135, r, *c = new vec3_t[w * h];
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP
  for (int y = 0; y<h; y++){                       // Loop over image rows
    fprintf(stderr,"\rRendering (%d spp) %5.2f%%", samps * 4, 100. * y/(h - 1));
    for (unsigned short x = 0, Xi[3] = {0, 0, y * y * y}; x<w; x++)   // Loop cols
      for (int sy = 0, i = (h - y - 1) * w + x; sy<2; sy++)     // 2x2 subpixel rows
        for (int sx = 0; sx<2; sx++, r = vec3_t()){        // 2x2 subpixel cols
          for (int s = 0; s<samps; s++){
            double r1 = 2 * erand48(Xi), dx = r1<1 ? sqrt(r1) - 1: 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi), dy = r2<1 ? sqrt(r2) - 1: 1 - sqrt(2 - r2);
            vec3_t d = cx * ( ( (sx + .5 + dx)/2 + x)/w - .5) +
                    cy * ( ( (sy + .5 + dy)/2 + y)/h - .5) + cam.d;
            r = r + radiance(ray_t(cam.o + d * 140, d.norm()), 0, Xi) * (1./samps);
          } // Camera rays are pushed ^^^^^ forward to start in interior
          c[i] = c[i] + vec3_t(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
        }
  }
  FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i = 0; i<w * h; i++) {
    int r, g, b;
    vec_to_rgb(c[i], r, g, b);
    fprintf(f,"%d %d %d ", r, g, b);
  }
}

