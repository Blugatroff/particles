pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}
use anyhow::*;
use std::{path::Path, num::NonZeroU32};
impl Texture {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: P,
    ) -> Result<Self> {
        // Needed to appease the borrow checker
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str();
        println!("{:?}", label);
        let img = match image::open(path) {
            Result::Ok(img) => img,
            Result::Err(_) => create_colored([255, 255, 255, 255]),
        };

        Ok(Self::from_image(device, queue, &img, label))
    }
    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Self {
        let format = match img {
            image::DynamicImage::ImageRgba8(_) => "rgba8",
            image::DynamicImage::ImageLuma8(_) => "luma8",
            image::DynamicImage::ImageLumaA8(_) => "lumaA8",
            image::DynamicImage::ImageLuma16(_) => "luma16",
            image::DynamicImage::ImageLumaA16(_) => "lumaA16",
            image::DynamicImage::ImageRgb8(_) => "rgb8",
            image::DynamicImage::ImageRgb16(_) => "rgba16",
            image::DynamicImage::ImageBgr8(_) => "bgr8",
            image::DynamicImage::ImageBgra8(_) => "bgra8",
            _ => "",
        };
        let start = std::time::Instant::now();
        let img = img.clone().into_rgba8();
        let dimensions = img.dimensions();
        let raw = img.into_raw();
        if format != "rgba8" {
            println!(
                "converting {} from {} to rgba8 took: {} seconds",
                label.unwrap_or(""),
                format,
                std::time::Instant::now()
                    .duration_since(start)
                    .as_secs_f32()
            );
        }
        let rgba = raw.as_slice();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new((4 * dimensions.0) as u32),
                rows_per_image: NonZeroU32::new((dimensions.1) as u32),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        };

        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}
#[allow(dead_code)]
pub fn create_colored(color: [u8; 4]) -> image::DynamicImage {
    let mut texture: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(2, 2);
    for pixel in texture.enumerate_pixels_mut() {
        *pixel.2 = image::Rgba(color);
    }
    image::DynamicImage::ImageRgba8(texture)
}
