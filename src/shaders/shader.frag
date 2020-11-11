#version 450

layout(location=0) in vec2 v_tex_coords;
layout(location=1) in float life_time;
layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

void main() {
    //f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
    //f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords) / 2 + texture(sampler2D(t_2, s_2), v_tex_coords) / 2;
    float l = life_time - 0.5;
    f_color = vec4(l, 1 - l, 0.3, 1);
}