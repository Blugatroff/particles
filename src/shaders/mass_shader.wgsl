struct VertexOutput {
    [[location(0)]] tex_coords: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[block]]
struct Uniforms {
    view_proj: mat4x4<f32>;
    model_transform: mat4x4<f32>;
};

[[group(1), binding(0)]]
var uniforms: Uniforms;

[[block]]
struct Instances {
    instances: [[stride(64)]] array<mat4x4<f32>>;
};

[[group(2), binding(0)]] var<storage, read> instances: Instances;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] tex_coords: vec2<f32>,
    [[builtin(instance_index)]] instance_index: u32,
) -> VertexOutput {
    var result: VertexOutput;
    result.tex_coords = tex_coords;
    result.position = uniforms.view_proj
        * instances.instances[instance_index]
        * uniforms.model_transform
        * vec4<f32>(position, 1.0);
    return result;
}

[[group(0), binding(0)]]
var texture: texture_2d<f32>;
[[group(0), binding(1)]]
var sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(texture, sampler, in.tex_coords);
}

