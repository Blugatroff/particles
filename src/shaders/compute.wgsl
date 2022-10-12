struct Uniforms {
    view_proj: mat4x4<f32>,
    n_masses: u32,
    n_particles_requested: u32,
    n_particles: u32,
    g: f32,
    delta_time: f32,
    push: f32,
    reset: u32,
    frame: u32,
    cam_pos: vec3<f32>,
    padding: u32,
    random: vec3<f32>,
};

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

struct Instance {
    position: vec3<f32>,
    scale: f32,
    velocity: vec3<f32>,
    padding: f32,
};

struct Instances {
    speeds: array<atomic<u32>, 2>,
    instances: array<Instance>,
}

@group(1)
@binding(0)
var<storage, read_write> instances: Instances;

struct Mass {
    position: vec3<f32>,
    scale: f32,
};

@group(2)
@binding(0)
var<storage, read> masses: array<Mass>;

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co.xy, vec2<f32>(12.9898,78.233))) * 43758.5453);
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= uniforms.n_particles_requested) {
        return;
    }
    if (id.x == 0u) {
        atomicStore(&instances.speeds[(uniforms.frame + 1u) % 2u], 0u);
    }
    let delta_time = min(uniforms.delta_time, 0.010);
    let frame = uniforms.frame;
    let instance: ptr<storage, Instance, read_write> = &instances.instances[id.x];
    if (id.x >= uniforms.n_particles) {
        let v = f32(id.x);
        (*instance).position = (uniforms.random - 0.5) + vec3<f32>(
            rand(vec2<f32>(v + delta_time + f32(frame), uniforms.random.x)),
            rand(vec2<f32>(v + delta_time + f32(frame), uniforms.random.y)),
            rand(vec2<f32>(v + delta_time + f32(frame), uniforms.random.z)),
        );
        (*instance).scale = 0.001;
        (*instance).velocity = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }
    if (uniforms.push > 0.0) {
        let delta = (*instance).position - uniforms.cam_pos;
        let strength = 1.0 / length(delta) * uniforms.push;
        (*instance).velocity += strength * normalize(delta);
        return;
    }
    if (uniforms.reset != 0u) {
        (*instance).velocity = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }
    let speed_atomic = &instances.speeds[uniforms.frame % 2u];
    let speed = u32(length((*instance).velocity) * 256.0);
    atomicAdd(speed_atomic, speed);

    let n_masses = uniforms.n_masses;
    let g = uniforms.g;
    var i: u32 = 0u;
    loop {
        if (i >= n_masses) {
            break;
        }
        let mass = masses[i];
        let delta: vec3<f32> = mass.position - (*instance).position;
        let scalar = (1.0 / length(delta))
            * 0.523598775598
            * pow(mass.scale, 3.0)
            * g
            * delta_time;
        let acceleration = scalar * delta;
        (*instance).velocity += acceleration * delta_time;
        i = i + 1u;
    }
    (*instance).velocity = (*instance).velocity * (1.0 - 0.01 * delta_time);
    (*instance).position += (*instance).velocity * delta_time;
}
