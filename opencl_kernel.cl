__constant float EPSILON = 0.00003f;
__constant float PI = 3.14159265358979323846f;
__constant int SAMPLES = 10;
__constant float inf = 1e20f;

typedef struct {
  float3 origin;
  float3 dir;
} Ray;

typedef struct {
  float radius;
  float3 pos;
  float3 color;
  float3 emission;
} Sphere;

uint wang_hash(uint seed) {
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

uint rand_xorshift(uint *rng_state) {
  *rng_state ^= (*rng_state << 13);
  *rng_state ^= (*rng_state >> 17);
  *rng_state ^= (*rng_state << 5);
  return *rng_state;
}

float random(uint *rng_state) {
  return (float)(rand_xorshift(rng_state)) * (1.0 / 4294967296.0);
}

Ray create_cam_ray(const int x_coord, const int y_coord, const int width
    , const int height) {
  // convert int in range [0 - width] to float in range [0-1]
  float fx = (float)x_coord / (float)width;
  float fy = (float)y_coord / (float)height;

  // calculate aspect ratio
  float aspect_ratio = (float)width / (float)height;
  float fx2 = (fx - 0.5f) * aspect_ratio;
  float fy2 = fy - 0.5f;

  // determine position of pixel on screen
  float3 pixel_pos = (float3)(fx2, fy2, 0.f);

  // create camera ray with fixed camera position and
  // vector from camera to pixel on screen
  Ray ray;
  ray.origin = (float3)(0.f, 0.1f, 2.f);
  ray.dir = normalize(pixel_pos - ray.origin);

  return ray;
}

// (__global Sphere* sphere, const Ray* ray)
// version using local copy of sphere
float intersect_sphere(const Sphere *sphere, const Ray *ray) {
  float3 ray_to_center = sphere->pos - ray->origin;
  float b = dot(ray_to_center, ray->dir);
  float c = dot(ray_to_center, ray_to_center) - sphere->radius * sphere->radius;
  float disc = b * b - c;

  if (disc < 0.f)
    return 0.f;
  else
    disc = sqrt(disc);

  if ((b - disc) > EPSILON)
    return b - disc;
  if ((b + disc) > EPSILON)
    return b + disc;

  return 0.f;
}

bool intersect_scene(__constant Sphere *spheres, const Ray *ray, float *t
    , int *sphere_id, const int num_spheres) {
  *t = inf;

  for (int i = 0; i < num_spheres; i++) {
    Sphere sphere = spheres[i]; // create local copy of sphere
    // float hitdistance = intersect_sphere(&spheres[i], ray);
    float hitdistance = intersect_sphere(&sphere, ray);
    // keep track of the closest intersection and hitobject found so far
    if (hitdistance != 0.f && hitdistance < *t) {
      *t = hitdistance;
      *sphere_id = i;
    }
  }
  return *t < inf; // true when ray interesects the scene
}

// the path tracing function
// computes a path (starting from the camera) with a defined number of bounces,
// accumulates light/color at each bounce. each ray hitting a surface will be
// reflected in a random direction (by randomly sampling the hemisphere above
// the hitpoint)
// small optimisation: diffuse ray directions are calculated using cosine
// weighted importance sampling
float3 trace(__constant Sphere *spheres, const Ray *camray
    , const int num_spheres, uint *rng_state) {
  Ray ray = *camray;

  float3 accum_color = (float3)(0.f, 0.f, 0.f);
  float3 mask = (float3)(1.f, 1.f, 1.f);

  for (int bounces = 0; bounces < 8; bounces++) {
    float t; // distance to intersection
    int hitsphere_id = 0; // index of intersected sphere

    // if ray misses scene, return background colour
    if (!intersect_scene(spheres, &ray, &t, &hitsphere_id, num_spheres))
      return accum_color += mask * (float3)(0.15f, 0.15f, 0.25f);

    // else, we've got a hit! Fetch the closest hit sphere
    // version with local copy of sphere
    Sphere hitsphere = spheres[hitsphere_id];

    // compute the hitpoint using the ray equation
    float3 hitpoint = ray.origin + ray.dir * t;

    // compute the surface normal and flip it if necessary to face the
    // incoming ray
    float3 normal = normalize(hitpoint - hitsphere.pos);
    float3 normal_facing = dot(normal, ray.dir) < 0.f ? normal : normal * (-1.f);

    // compute two random numbers to pick a random point on the hemisphere above
    // the hitpoint
    float rand1 = 2.f * PI * random(rng_state);
    float rand2 = random(rng_state);
    float rand2s = sqrt(rand2);

    // create a local orthogonal coordinate frame centered at the hitpoint
    float3 w = normal_facing;
    float3 axis = fabs(w.x) > EPSILON ? (float3)(0.f, 1.f, 0.f)
      : (float3)(1.f, 0.f, 0.f);
    float3 u = normalize(cross(axis, w));
    float3 v = cross(w, u);

    // use the coordinte frame and random numbers to compute the next ray
    // direction
    float3 newdir = normalize(u * cos(rand1) * rand2s + v * sin(rand1) * rand2s
        + w * sqrt(1.f - rand2));

    // add a very small offset to the hitpoint to prevent self intersection
    ray.origin = hitpoint + normal_facing * EPSILON;
    ray.dir = newdir;

    // add the colour and light contributions to the accumulated colour
    accum_color += mask * hitsphere.emission;

    // the mask colour picks up surface colours at each bounce
    mask *= hitsphere.color;

    // perform cosine-weighted importance sampling for diffuse surfaces
    mask *= dot(newdir, normal_facing);

#if 0
    // R.R.
    if (bounces > 3) {
      float p = max(mask.x, max(mask.y, mask.z));
      if (random(rng_state) > p)
        break;
      mask *= 1.f / p;
    }
#endif
  }

  return accum_color;
}

float linear_to_srgb(float x) {
  if (x < 0.0031308f)
    x *= 12.92f;
  else
    x = 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
  return x;
}

float linear_to_srgb_clamp(float x) {
  return linear_to_srgb(clamp(x, 0.f, 1.f));
}

float4 linear_to_srgb_clamp4(float3 c) {
  return (float4)(linear_to_srgb_clamp(c.x), linear_to_srgb_clamp(c.y)
      , linear_to_srgb_clamp(c.z), 1.f);
}

__kernel void render_kernel(__constant Sphere *spheres, const int num_spheres,
    write_only image2d_t out, const int width, const int height) {
  // the unique global id of the work item for the current pixel
  unsigned int work_item_id = get_global_id(0);

  uint rng_state = wang_hash(work_item_id);

  unsigned int x_coord = work_item_id % width;
  unsigned int y_coord = work_item_id / width;

  if (x_coord >= width || y_coord >= height)
    return;

  Ray camray = create_cam_ray(x_coord, y_coord, width, height);

  // add the light contribution of each sample and average over all samples
  float3 finalcolor = (float3)(0.f, 0.f, 0.f);
  float inv_samples = 1.f / SAMPLES;

  for (int i = 0; i < SAMPLES; i++)
    finalcolor += trace(spheres, &camray, num_spheres, &rng_state) * inv_samples;

  write_imagef(out, (int2)(x_coord, y_coord), linear_to_srgb_clamp4(finalcolor));
}

