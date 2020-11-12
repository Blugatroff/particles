const SCALE: f32 = 0.001;
const RANGE: i32 = 10;
const DRAG: f32 = 0.01;
const BOOMPOWER: f32 = 1.0;
const MAXFRAMETIME: f32 = 1.0 / 60.0;

use super::*;
use anyhow::*;
use model::*;
use rayon::prelude::*;
use winit::window::Window;

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
    fn to_raw(&self) -> InstanceRaw {
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
    fn refresh(&mut self, device: &wgpu::Device) {
        let data: Vec<InstanceRaw> = self
            .instances
            .iter()
            .map(|instance| instance.to_raw())
            .collect();
        self.binding.buffer =
            device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                });
        self.binding.bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.binding.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        self.binding.buffer.slice(..),
                    ),
                }],
                label: Some("instance_bind_group"),
            });
    }
}

#[allow(dead_code)]
pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
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
}
impl State {
    pub async fn new(window: &Window, particle_number: i32, g: f32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();

        let window_size = window.inner_size();
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Immediate,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let camera = Camera::new(sc_desc.width as f32 / sc_desc.height as f32);

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
            label: Some("uniform_bind_group"),
        });
        let uniform_binding = Binding {
            buffer: uniform_buffer,
            bind_group_layout: uniform_bind_group_layout,
            bind_group: uniform_bind_group,
        };
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
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
        let instance_data: Vec<InstanceRaw> =
            instances.iter().map(ParticleInstance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        let instance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("instance_bind_group_layout"),
            });

        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(instance_buffer.slice(..)),
            }],
            label: Some("instance_bind_group"),
        });
        let instance_data_data = instances.iter().map(ParticleInstance::to_data).collect::<Vec<_>>();
        let instance_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data_data),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });
        let instances_binding = Binding {
            buffer: instance_buffer,
            bind_group_layout: instance_bind_group_layout,
            bind_group: instance_bind_group,
        };

        let instance_data_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("instance_data_bind_group_layout"),
            });

        let instance_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instance_data_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(instance_data_buffer.slice(..)),
            }],
            label: Some("instance_data_bind_group"),
        });
        let instance_data_binding = Binding {
            buffer: instance_data_buffer,
            bind_group_layout: instance_data_bind_group_layout,
            bind_group: instance_data_bind_group,
        };

        let vs_module =
            device.create_shader_module(wgpu::include_spirv!("shaders/shader.vert.spv"));
        let fs_module =
            device.create_shader_module(wgpu::include_spirv!("shaders/shader.frag.spv"));

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

        let render_pipeline = create_render_pipeline(
            &device,
            render_pipeline_layout,
            vs_module,
            fs_module,
            &sc_desc,
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
                scale: 5 as f32 / 100.0,
                //velocity: cgmath::Vector3::new(0.0, 0.0, 0.0),
            });
        }
        let mass_instances_data = mass_instances
            .iter()
            .map(MassInstance::to_raw)
            .collect::<Vec<_>>();
        let mass_instances_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&mass_instances_data),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });
        let mass_instances_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("instance_bind_group_layout"),
            });
        let mass_instances_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &mass_instances_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(mass_instances_buffer.slice(..)),
            }],
            label: Some("instance_bind_group"),
        });
        let mass_instances_binding = Binding {
            buffer: mass_instances_buffer,
            bind_group_layout: mass_instances_bind_group_layout,
            bind_group: mass_instances_bind_group,
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
            device.create_shader_module(wgpu::include_spirv!("shaders/mass_shader.vert.spv"));
        let fs_module =
            device.create_shader_module(wgpu::include_spirv!("shaders/mass_shader.frag.spv"));
        let mass_pipeline =
            create_render_pipeline(&device, mass_pipe_layout, vs_module, fs_module, &sc_desc);
        Ok(Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            particles: PipePackage {
                pipeline: render_pipeline,
                model,
                binding: instances_binding,
                data_binding: instance_data_binding,
                instances: instances,
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
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
        self.camera.aspect = (new_size.width as f32) / (new_size.height as f32);
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.sc_desc, "depth_texture");
    }
    pub fn device_input(&mut self, event: &DeviceEvent) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.camera.controller.heading +=
                    delta.0 as f32 * -self.camera.controller.mouse_sensitivity;
                self.camera.controller.pitch +=
                    delta.1 as f32 * -self.camera.controller.mouse_sensitivity;
            }
            _ => {}
        }
    }
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_events(event);
        match event {
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    virtual_keycode,
                    state,
                    ..
                } => match (state, virtual_keycode) {
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
                    _ => {}
                },
            },
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
        self.queue.write_buffer(
            &self.uniform_binding.buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        println!(
            "{}, particle_num: {}, frametime: {}",
            if delta_time < MAXFRAMETIME { "+" } else { "-" },
            self.particles.instances.len(),
            delta_time,
        );
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
            self.particle_adding_value = if self.particle_adding_value > 0 { 0 } else { self.particle_adding_value };
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
        self.test_angle += 1.0 / 3.14159 * delta_time;

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
                        * (0.5236 * mass.scale.powf(3.0))
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

        self.particles.refresh(&self.device);

        let instance_data_data: Vec<DataRaw> = self
            .particles
            .instances
            .iter()
            .map(|instance| instance.to_data())
            .collect();
        self.particles.data_binding.buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instance_data_data),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                });
        self.particles.data_binding.bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.particles.data_binding.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        self.particles.data_binding.buffer.slice(..),
                    ),
                }],
                label: Some("instance_bind_group"),
            });

        let mass_instance_data: Vec<InstanceRaw> = self
            .masses.instances
            .iter()
            .map(|instance| instance.to_raw())
            .collect();
        self.queue.write_buffer(
            &self.masses.binding.buffer,
            0,
            bytemuck::cast_slice(&mass_instance_data),
        );
    }

    pub fn render(&mut self) {
        let frame = self
            .swap_chain
            .get_current_frame()
            .expect("Timeout getting texture")
            .output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.particles.pipeline);
            for mesh in &self.particles.model.meshes {
                let material = &self.particles.model.materials[mesh.material];
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..));
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
                render_pass.set_index_buffer(mesh.index_buffer.slice(..));
                render_pass.set_bind_group(0, &material.bind_group, &[]);
                render_pass.set_bind_group(1, &self.uniform_binding.bind_group, &[]);
                render_pass.set_bind_group(2, &self.masses.binding.bind_group, &[]);
                render_pass.draw_indexed(
                    0..mesh.num_elements,
                    0,
                    0..self.particles.instances.len() as u32,
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
                /* instance.position.x = 0.0;
                instance.position.y = 0.0;
                instance.position.z = 0.0; */
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
    sc_desc: &wgpu::SwapChainDescriptor,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&layout),
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
            clamp_depth: false,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: sc_desc.format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: texture::Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilStateDescriptor::default(),
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint32,
            vertex_buffers: &[Vertex::desc()],
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
    })
}
