struct VertexOutput {
    @location(0) tex_coords: vec2<f32>,
    @location(1) life_time: f32,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(1)
@binding(0)
var<uniform> uniforms: Uniforms;

struct Instance {
    position: vec3<f32>,
    scale: f32,
    life_time: f32,
};

struct Instances {
    instances: array<Instance>,
};

@group(2)
@binding(0)
var<storage, read> instances: Instances;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var result: VertexOutput;
    result.tex_coords = tex_coords;
    var instance = instances.instances[instance_index];
    result.life_time = instance.life_time;
    result.position = uniforms.view_proj * vec4<f32>(instance.position + instance.scale * position, 1.0);
    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let l = in.life_time - 0.5;
    return vec4<f32>(l, 1.0 - l, 0.3, 1.0);
}
