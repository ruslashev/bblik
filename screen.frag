#version 330

uniform sampler2D tex;
in vec2 texcoord_f;

out vec4 frag_color;

void main() {
  frag_color = texture2D(tex, texcoord_f);
}

