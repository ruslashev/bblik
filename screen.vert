#version 330

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 matrix;

out vec2 texcoord_f;

void main() {
  texcoord_f = texcoord;
  gl_Position = matrix * vec4(position, 0.0, 1.0);
}

