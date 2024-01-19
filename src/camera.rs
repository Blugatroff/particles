use winit::keyboard::{PhysicalKey, KeyCode};

use super::*;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub controller: CameraController,
}

pub struct CameraController {
    pub speed: f32,
    pub right: bool,
    pub left: bool,
    pub forward: bool,
    pub backward: bool,
    pub down: bool,
    pub up: bool,
    pub look_right: bool,
    pub look_left: bool,
    pub look_down: bool,
    pub look_up: bool,
    pub heading: f32,
    pub pitch: f32,
    pub mouse_sensitivity: f32,
    pub key_sensitivity: f32,
}
impl Camera {
    pub fn new(aspect: f32) -> Self {
        Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 0.0, -2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect,
            fovy: 75.0,
            znear: 0.1,
            zfar: 100.0,
            controller: CameraController {
                speed: 1.0,
                right: false,
                left: false,
                forward: false,
                backward: false,
                down: false,
                up: false,
                look_right: false,
                look_left: false,
                look_down: false,
                look_up: false,
                heading: 0.0,
                pitch: 0.0,
                mouse_sensitivity: 0.0025,
                key_sensitivity: 1.0,
            },
        }
    }
    pub fn reset(&mut self) {
        self.eye = (0.0, 0.0, -2.0).into();
        self.target = (0.0, 0.0, 0.0).into();
    }
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
    #[rustfmt::skip]
    pub fn update(&mut self, delta_time: f32) {
        if self.controller.up   { self.eye.y += self.controller.speed * delta_time }
        if self.controller.down { self.eye.y -= self.controller.speed * delta_time }

        if self.controller.look_right { self.controller.heading += self.controller.key_sensitivity * delta_time ; }
        if self.controller.look_left  { self.controller.heading -= self.controller.key_sensitivity * delta_time ; }
        if self.controller.look_down  { self.controller.pitch   -= self.controller.key_sensitivity * delta_time ; }
        if self.controller.look_up    { self.controller.pitch   += self.controller.key_sensitivity * delta_time ; }

        self.controller.pitch = self.controller.pitch.clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI  / 2.0);

        let direction = cgmath::Vector3::new(
            self.controller.pitch.cos() * self.controller.heading.sin(),
            self.controller.pitch.sin(),
            self.controller.pitch.cos() * self.controller.heading.cos(),
        );
        let plane_direction = cgmath::Vector3::new(
            direction.x,
            0.0,
            direction.z,
        ).normalize();
        let right: cgmath::Vector3<f32> = cgmath::Vector3::new(
            (self.controller.heading - std::f32::consts::PI / 2.0).sin(),
            0.0,
            (self.controller.heading - std::f32::consts::PI / 2.0).cos()
        ).normalize();

        if self.controller.forward {
            self.eye += plane_direction * delta_time * self.controller.speed;
        }
        if self.controller.backward {
            self.eye -= plane_direction * delta_time * self.controller.speed;
        }
        if self.controller.right {
            self.eye += right * delta_time * self.controller.speed;
        }
        if self.controller.left {
            self.eye -= right * delta_time * self.controller.speed;
        }

        self.target = self.eye + direction;
    }
    #[rustfmt::skip]
    pub fn process_events(&mut self, event: &WindowEvent){
        if let WindowEvent::KeyboardInput { event, .. } = event { 
            match (event.state, event.physical_key) {
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::KeyD))  => { self.controller.right      = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::KeyD))  => { self.controller.right      = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::KeyA))  => { self.controller.left       = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::KeyA))  => { self.controller.left       = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ControlLeft))  => { self.controller.down       = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ControlLeft))  => { self.controller.down       = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::Space))  => { self.controller.up         = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::Space))  => { self.controller.up         = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::KeyW))  => { self.controller.forward    = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::KeyW))  => { self.controller.forward    = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::KeyS))  => { self.controller.backward   = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::KeyS))  => { self.controller.backward   = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ArrowLeft))  => { self.controller.look_right = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ArrowLeft))  => { self.controller.look_right = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ArrowRight))  => { self.controller.look_left  = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ArrowRight))  => { self.controller.look_left  = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ArrowDown))  => { self.controller.look_down  = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ArrowDown))  => { self.controller.look_down  = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ArrowUp))  => { self.controller.look_up    = true ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ArrowUp))  => { self.controller.look_up    = false; }
                (ElementState::Pressed , PhysicalKey::Code(KeyCode::ShiftLeft))  => { self.controller.speed     *= 2.0  ; }
                (ElementState::Released, PhysicalKey::Code(KeyCode::ShiftLeft))  => { self.controller.speed     /= 2.0  ; }
                _ => {}
            };
        }
    }
}
