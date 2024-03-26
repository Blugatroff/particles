use futures::executor::block_on;
use wgpu::util::DeviceExt;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};
mod camera;
use camera::*;
mod state;
mod texture;
use anyhow::*;
use cgmath::prelude::*;
use cgmath::SquareMatrix;
use state::*;
use std::{env, sync::Arc};
mod model;
fn main() -> Result<()> {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let mut is_focused = true;
    let num = {
        if args.len() > 1 {
            let num = args[1].parse::<i32>();
            println!("{num:?}");
            num.unwrap_or(1000)
        } else {
            1000
        }
    };
    let g = {
        if args.len() > 2 {
            let g = args[2].parse::<f32>();
            println!("{g:?}");
            g.unwrap_or(1.0)
        } else {
            1.0
        }
    };
    // Since main can't be async, we're going to need to block
    let mut state = block_on(State::new(window.clone(), num, g * 100000.0))?;
    window
        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .ok();
    window.set_cursor_visible(false);
    window.set_title("particles");
    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::Focused(true) => {
                        window.set_cursor_visible(false);
                        window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .ok();
                        is_focused = true;
                    }
                    WindowEvent::Focused(false) => {
                        window.set_cursor_visible(true);
                        window
                            .set_cursor_grab(winit::window::CursorGrabMode::None)
                            .unwrap();
                        is_focused = false;
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        state.render();
                    }
                    _ => {}
                }
            }
        }
        Event::DeviceEvent { event, .. } => {
            if is_focused {
                state.device_input(&event);
            }
        }
        Event::AboutToWait => {
            window.request_redraw();
        }
        _ => {}
    })?;
    Ok(())
}
