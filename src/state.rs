const SCALE: f32 = 0.001;
const RANGE: i32 = 10;
const DRAG: f32 = 0.01;
const BOOMPOWER: f32 = 1.0;
const MAXFRAMETIME: f32 = 1.0 / 60.0;

//use std::num::NonZeroU64;

use std::num::NonZeroU64;

use super::*;
use anyhow::*;
use model::*;
use rayon::prelude::*;
use winit::window::Window;

fn align_to(n: u64, alignment: u64) -> u64 {
    n + (n % alignment)
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Uniforms {
    view_proj: cgmath::Matrix4<f32>,
    model_transform: cgmath::Matrix4<f32>,
}

unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity(),
            model_transform: cgmath::Matrix4::identity(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct InstanceRaw {
    model: cgmath::Matrix4<f32>,
}

unsafe impl bytemuck::Pod for InstanceRaw {}
unsafe impl bytemuck::Zeroable for InstanceRaw {}
#[repr(C)]
#[derive(Copy, Clone)]
struct DataRaw {
    life_time: f32,
}
unsafe impl bytemuck::Pod for DataRaw {}
unsafe impl bytemuck::Zeroable for DataRaw {}

struct ParticleInstance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    life_time: f32,
    velocity: cgmath::Vector3<f32>,
    scale: f32,
}
impl ParticleInstance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from_scale(self.scale)
                * cgmath::Matrix4::from(self.rotation),
        }
    }
    fn to_data(&self) -> DataRaw {
        DataRaw {
            life_time: self.life_time,
        }
    }
}

#[derive(Copy, Clone)]
struct MassInstance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: f32,
}
impl MassInstance {
    fn into_raw(self) -> InstanceRaw {
        InstanceRaw {
            /* life_time: self.life_time, */
            model: cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from_scale(self.scale)
                * cgmath::Matrix4::from(self.rotation),
        }
    }
}

struct Binding {
    buffer: wgpu::Buffer,
    buffer_size: u64,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

struct PipePackage {
    pipeline: wgpu::RenderPipeline,
    model: Model,
    binding: Binding,
    data_binding: Binding,
    instances: Vec<ParticleInstance>,
}

struct PipePackageMass {
    pipeline: wgpu::RenderPipeline,
    model: Model,
    binding: Binding,
    instances: Vec<MassInstance>,
}

impl PipePackage {
    fn refresh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let data: Vec<InstanceRaw> = self
            .instances
            .iter()
            .map(|instance| instance.to_raw())
            .collect();
        let data_size = (data.len() * std::mem::size_of::<InstanceRaw>()) as u64;
        if data_size > self.binding.buffer_size {
            let buffer_size = align_to(data_size * 3 / 2, 4);
            self.binding.buffer_size = buffer_size;
            self.binding.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.binding.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.binding.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.binding.buffer,
                        offset: 0,
                        size: NonZeroU64::new(self.binding.buffer_size),
                    }),
                }],
                label: Some("instance_bind_group"),
            });
        }
        unsafe {
            queue.write_buffer(
                &self.binding.buffer,
                0,
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data_size as usize),
            );
        }
        let data: Vec<DataRaw> = self
            .instances
            .iter()
            .map(|instance| instance.to_data())
            .collect();
        let data_size = (data.len() * std::mem::size_of::<DataRaw>()) as u64;
        if data_size > self.data_binding.buffer_size {
            let buffer_size = align_to(data_size * 3 / 2, 4);
            self.data_binding.buffer_size = buffer_size;

            self.data_binding.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Instance Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.data_binding.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.data_binding.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.data_binding.buffer,
                        offset: 0,
                        size: NonZeroU64::new(self.data_binding.buffer_size),
                    }),
                }],
                label: Some("instance_bind_group"),
            });
        }
        unsafe {
            queue.write_buffer(
                &self.data_binding.buffer,
                0,
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data_size as usize),
            );
        }
    }
}

#[allow(dead_code)]
pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    depth_texture: texture::Texture,
    uniforms: Uniforms,
    uniform_binding: Binding,
    particles: PipePackage,
    masses: PipePackageMass,
    camera: Camera,
    last_time: std::time::Instant,
    test_angle: f32,
    avg_speed_last_frame: f32,
    g: f32,
    particle_adding_value: i32,
    print: bool,
}
impl State {
    pub async fn new(window: &Window, particle_number: i32, g: f32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let window_size = window.inner_size();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Immediate,
        };
        surface.configure(&device, &surface_config);

        let camera = Camera::new(surface_config.width as f32 / surface_config.height as f32);

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
            label: Some("uniform_bind_group"),
        });
        let uniform_binding = Binding {
            buffer: uniform_buffer,
            bind_group_layout: uniform_bind_group_layout,
            bind_group: uniform_bind_group,
            buffer_size: std::mem::size_of::<Uniforms>() as u64,
        };
        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            surface_config.width,
            surface_config.height,
            "depth_texture",
        );
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            "assets/simpleParticle.obj",
        )?;
        let mut instances: Vec<ParticleInstance> = Vec::new();
        for _ in 0..particle_number {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let x: f32 = rng.gen::<f32>() % RANGE as f32 - 0.5;
            let y: f32 = rng.gen::<f32>() % RANGE as f32 - 0.5;
            let z: f32 = rng.gen::<f32>() % RANGE as f32 - 0.5;
            let life_time = rng.gen::<f32>() % 1.0;
            instances.push(ParticleInstance {
                position: cgmath::Vector3::new(x, y, z),
                rotation: cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::new(0.0, 1.0, 0.0),
                    cgmath::Deg(0.1),
                ),
                scale: SCALE,
                life_time,
                velocity: cgmath::Vector3::new(0.0, 0.0, 0.0),
            });
        }
        let instance_data: Vec<InstanceRaw> = instances
            .iter()
            .map(ParticleInstance::to_raw)
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let instance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("instance_bind_group_layout"),
            });

        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &instance_buffer,
                    offset: 0,
                    size: NonZeroU64::new(
                        (instance_data.len() * std::mem::size_of::<InstanceRaw>()) as u64,
                    ),
                }),
            }],
            label: Some("instance_bind_group"),
        });
        let instance_data_data = instances
            .iter()
            .map(ParticleInstance::to_data)
            .collect::<Vec<_>>();
        let instance_data_size = instance_data_data.len() * std::mem::size_of::<DataRaw>();
        let instance_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let instances_binding = Binding {
            buffer: instance_buffer,
            bind_group_layout: instance_bind_group_layout,
            bind_group: instance_bind_group,
            buffer_size: (instance_data.len() * std::mem::size_of::<InstanceRaw>()) as u64,
        };

        let instance_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("instance_data_bind_group_layout"),
            });

        let instance_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instance_data_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &instance_data_buffer,
                    offset: 0,
                    size: NonZeroU64::new(instance_data_size as u64),
                }),
            }],
            label: Some("instance_data_bind_group"),
        });
        let instance_data_binding = Binding {
            buffer: instance_data_buffer,
            bind_group_layout: instance_data_bind_group_layout,
            bind_group: instance_data_bind_group,
            buffer_size: instance_data_size as u64,
        };

        let vs_module =
            device.create_shader_module(&wgpu::include_spirv!("shaders/particle_shader.vert.spv"));
        let fs_module =
            device.create_shader_module(&wgpu::include_spirv!("shaders/particle_shader.frag.spv"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &uniform_binding.bind_group_layout,
                    &instances_binding.bind_group_layout,
                    &instance_data_binding.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let vertex_buffer_layouts = &[Vertex::desc()];
        let color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }];
        let render_pipeline = create_render_pipeline(
            &device,
            render_pipeline_layout,
            vs_module,
            fs_module,
            vertex_buffer_layouts,
            color_targets,
            Some("Particle Pipeline"),
        );

        let mass = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            "assets/sphere.obj",
        )?;
        let mass_num = 5;
        let mut mass_instances: Vec<MassInstance> = Vec::new();
        for i in 0..mass_num {
            mass_instances.push(MassInstance {
                position: cgmath::Vector3::new(i as f32, 0.0, 0.0),
                rotation: cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::new(0.0, 1.0, 0.0),
                    cgmath::Deg(0.0),
                ),
                scale: 5.0 / 100.0,
            });
        }
        let mass_instances_data = mass_instances
            .iter()
            .copied()
            .map(MassInstance::into_raw)
            .collect::<Vec<_>>();
        let mass_instance_size = mass_instances_data.len() * std::mem::size_of::<InstanceRaw>();
        let mass_instances_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&mass_instances_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mass_instances_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                    },
                    count: None,
                }],
                label: Some("instance_bind_group_layout"),
            });
        let mass_instances_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &mass_instances_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &mass_instances_buffer,
                    size: NonZeroU64::new(mass_instance_size as u64),
                    offset: 0,
                }),
            }],
            label: Some("instance_bind_group"),
        });
        let mass_instances_binding = Binding {
            buffer: mass_instances_buffer,
            bind_group_layout: mass_instances_bind_group_layout,
            bind_group: mass_instances_bind_group,
            buffer_size: mass_instance_size as u64,
        };
        let mass_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &uniform_binding.bind_group_layout,
                &instances_binding.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let vs_module =
            device.create_shader_module(&wgpu::include_spirv!("shaders/mass_shader.vert.spv"));
        let fs_module =
            device.create_shader_module(&wgpu::include_spirv!("shaders/mass_shader.frag.spv"));
        let vertex_buffer_layouts = &[Vertex::desc()];
        let color_targets = &[wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }];
        let mass_pipeline = create_render_pipeline(
            &device,
            mass_pipe_layout,
            vs_module,
            fs_module,
            vertex_buffer_layouts,
            color_targets,
            Some("MASS PIPELINE"),
        );
        Ok(Self {
            print: false,
            surface,
            device,
            queue,
            surface_config,
            particles: PipePackage {
                pipeline: render_pipeline,
                model,
                binding: instances_binding,
                data_binding: instance_data_binding,
                instances,
            },
            camera,
            uniforms,
            uniform_binding,
            masses: PipePackageMass {
                pipeline: mass_pipeline,
                model: mass,
                binding: mass_instances_binding,
                instances: mass_instances,
            },
            depth_texture,
            last_time: std::time::Instant::now(),
            test_angle: 0.0,
            avg_speed_last_frame: 0.0,
            g,
            particle_adding_value: 0,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.camera.aspect = (new_size.width as f32) / (new_size.height as f32);
        self.depth_texture = texture::Texture::create_depth_texture(
            &self.device,
            new_size.width,
            new_size.height,
            "depth_texture",
        );
    }
    pub fn device_input(&mut self, event: &DeviceEvent) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            self.camera.controller.heading +=
                delta.0 as f32 * -self.camera.controller.mouse_sensitivity;
            self.camera.controller.pitch +=
                delta.1 as f32 * -self.camera.controller.mouse_sensitivity;
        }
    }
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_events(event);
        match event {
            WindowEvent::KeyboardInput { input, .. } => {
                let KeyboardInput {
                    virtual_keycode,
                    state,
                    ..
                } = input;
                match (state, virtual_keycode) {
                    (ElementState::Pressed, Some(VirtualKeyCode::R)) => {
                        self.reset();
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::C)) => {
                        self.camera.reset();
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::Add)) => {
                        self.g *= 1.05;
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::Subtract)) => {
                        self.g *= 0.95;
                    }
                    (ElementState::Pressed, Some(VirtualKeyCode::B)) => {
                        let eye = self.camera.eye;
                        self.particles
                            .instances
                            .par_iter_mut()
                            .for_each(|instance| {
                                let x = eye.x;
                                let y = eye.y;
                                let z = eye.z;
                                let delta: cgmath::Vector3<f32> =
                                    cgmath::Vector3::new(x, y, z) - instance.position;
                                let scalar = (1.0 / delta.magnitude()) * -BOOMPOWER;
                                let acceleration = cgmath::Vector3 {
                                    x: delta.x * scalar,
                                    y: delta.y * scalar,
                                    z: delta.z * scalar,
                                };
                                instance.velocity += acceleration;
                            });
                    }
                    (state, Some(VirtualKeyCode::P)) => {
                        self.print = *state == ElementState::Pressed;
                    }
                    _ => {}
                };
            }
            _ => return false,
        }
        true
    }
    pub fn update(&mut self) {
        let delta_time = std::time::Instant::now()
            .duration_since(self.last_time)
            .as_secs_f32();
        self.last_time = std::time::Instant::now();
        self.camera.update(delta_time);
        self.uniforms.update_view_proj(&self.camera);
        let data = &[self.uniforms];
        let data = bytemuck::cast_slice(data);
        self.queue
            .write_buffer(&self.uniform_binding.buffer, 0, data);
        if self.print {
            println!(
                "{}, particle_num: {}, frametime: {}, pos: {:?}",
                if delta_time < MAXFRAMETIME { "+" } else { "-" },
                self.particles.instances.len(),
                delta_time,
                self.camera.eye
            );
        }
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let init_vel = 0.5;
        if delta_time < MAXFRAMETIME {
            self.particle_adding_value += 1;
            for _ in 0..self.particle_adding_value {
                self.particles.instances.push(ParticleInstance {
                    position: cgmath::Vector3::new(0.0, 0.3, 0.0),
                    rotation: cgmath::Quaternion::from_axis_angle(
                        cgmath::Vector3::new(0.0, 1.0, 0.0),
                        cgmath::Deg(rng.gen::<f32>() % 360.0),
                    ),
                    scale: SCALE,
                    life_time: 1.0,
                    velocity: cgmath::Vector3::new(
                        (rng.gen::<f32>() % init_vel) - (init_vel / 2.0),
                        (rng.gen::<f32>() % init_vel) - (init_vel / 2.0),
                        (rng.gen::<f32>() % init_vel) - (init_vel / 2.0),
                    ),
                })
            }
        } else {
            self.particle_adding_value = if self.particle_adding_value > 0 {
                0
            } else {
                self.particle_adding_value
            };
            self.particle_adding_value -= 1;
            for _ in 0..self.particle_adding_value {
                self.particles.instances.pop();
            }
        }

        let l: f32 = 1.0;
        for i in 0..self.masses.instances.len() {
            self.masses.instances[i].position.x =
                (self.test_angle / (i + 1) as f32).cos() * l * i as f32;
            self.masses.instances[i].position.z =
                (self.test_angle / (i + 1) as f32).sin() * l * i as f32;
        }
        self.test_angle += 1.0 / std::f32::consts::PI * delta_time;

        let mut sum = 0.0;
        let mass_instances = &self.masses.instances;
        let g = self.g;
        self.particles
            .instances
            .par_iter_mut()
            .for_each(|instance| {
                for mass in mass_instances {
                    let delta: cgmath::Vector3<f32> = mass.position - instance.position;
                    let scalar = (1.0 / delta.magnitude())
                        * (std::f32::consts::FRAC_PI_6 * mass.scale.powf(3.0))
                        * g
                        * delta_time;
                    let acceleration = cgmath::Vector3 {
                        x: delta.x * scalar,
                        y: delta.y * scalar,
                        z: delta.z * scalar,
                    };
                    instance.velocity += acceleration * delta_time;
                }
                instance.velocity *= 1.0 - DRAG * delta_time;
                instance.position += instance.velocity * delta_time;
            });
        let avg_speed_last_frame = self.avg_speed_last_frame;
        self.particles.instances.iter_mut().for_each(|instance| {
            let speed: f32 = instance.velocity.magnitude();
            instance.life_time = speed / avg_speed_last_frame;
            sum += speed;
        });
        self.avg_speed_last_frame = sum / self.particles.instances.len() as f32;

        self.particles.refresh(&self.device, &self.queue);
        let mass_instance_data: Vec<InstanceRaw> = self
            .masses
            .instances
            .iter()
            .map(|instance| instance.into_raw())
            .collect();
        let data = bytemuck::cast_slice(&mass_instance_data);
        self.queue
            .write_buffer(&self.masses.binding.buffer, 0, data);
    }

    pub fn render(&mut self) {
        let frame = self
            .surface
            .get_current_frame()
            .expect("Timeout getting texture")
            .output;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
                label: None,
            });

            render_pass.set_pipeline(&self.particles.pipeline);
            for mesh in &self.particles.model.meshes {
                let material = &self.particles.model.materials[mesh.material];
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &material.bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_binding.bind_group, &[]);
                render_pass.set_bind_group(2, &self.particles.binding.bind_group, &[]);
                render_pass.set_bind_group(3, &self.particles.data_binding.bind_group, &[]);
                render_pass.draw_indexed(
                    0..mesh.num_elements,
                    0,
                    0..self.particles.instances.len() as u32,
                );
            }
            render_pass.set_pipeline(&self.masses.pipeline);
            for mesh in &self.masses.model.meshes {
                let material = &self.masses.model.materials[mesh.material];
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.set_bind_group(0, &material.bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_binding.bind_group, &[]);
                render_pass.set_bind_group(2, &self.masses.binding.bind_group, &[]);
                render_pass.draw_indexed(
                    0..mesh.num_elements,
                    0,
                    0..self.masses.instances.len() as u32,
                );
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
    fn reset(&mut self) {
        self.particles
            .instances
            .par_iter_mut()
            .for_each(|instance| {
                instance.velocity.x = 0.0;
                instance.velocity.y = 0.0;
                instance.velocity.z = 0.0;
            });
    }
}
fn create_render_pipeline(
    device: &wgpu::Device,
    layout: wgpu::PipelineLayout,
    vs_module: wgpu::ShaderModule,
    fs_module: wgpu::ShaderModule,
    vertex_buffer_layouts: &[wgpu::VertexBufferLayout],
    color_targets: &[wgpu::ColorTargetState],
    label: Option<&'static str>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label,
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "main",
            buffers: vertex_buffer_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            targets: color_targets,
            module: &fs_module,
            entry_point: "main",
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            clamp_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    })
}
