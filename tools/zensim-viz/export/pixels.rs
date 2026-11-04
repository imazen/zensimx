//! Imazen PNG/JPEG ingress, adapted from zensim/examples/support/zen_io.rs.
//! RGB8 research interpretation; higher-precision inputs are refused;
//! opaque inputs only. No browser or third-party decoder enters this path.
use crate::Result;
use std::path::Path;
pub(crate) struct RgbImage {
    pub pixels: Vec<[u8; 3]>,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
}
pub(crate) fn decode(path: &Path) -> Result<RgbImage> {
    let bytes = std::fs::read(path)?;
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "png" => {
            use zenpixels::ChannelType;
            let out = zenpng::decode(
                &bytes,
                &zenpng::PngDecodeConfig::default().with_max_pixels(16_000_000),
                &enough::Unstoppable,
            )?;
            let (width, height) = (out.info.width as usize, out.info.height as usize);
            let desc = out.pixels.descriptor();
            let channels = desc.channels() as usize;
            if desc.has_alpha() {
                return Err("atlas PNG ingress currently requires opaque RGB/gray input without an alpha channel".into());
            }
            let slice = out.pixels.as_slice();
            let mut pixels = Vec::new();
            for y in 0..height {
                let row = slice.row(y as u32);
                let samples = match desc.channel_type() {
                    ChannelType::U8 => row[..width * channels].to_vec(),
                    ChannelType::U16 => return Err("16-bit PNG requires a native-precision atlas path; RGB8 ingress refuses downconversion".into()),
                    _ => return Err("unsupported PNG channel type".into()),
                };
                for x in 0..width {
                    pixels.push(match channels {
                        1 => [samples[x]; 3],
                        3 => [samples[x * 3], samples[x * 3 + 1], samples[x * 3 + 2]],
                        _ => return Err("unsupported PNG layout".into()),
                    });
                }
            }
            Ok(RgbImage {
                pixels,
                width,
                height,
                stride: width,
            })
        }
        "jpg" | "jpeg" => {
            let out = zenjpeg::decoder::Decoder::new().decode(&bytes, enough::Unstoppable)?;
            let (width, height) = out.dimensions();
            let p = out.pixels_u8().ok_or("JPEG decoder did not return RGB8")?;
            if p.len() != width as usize * height as usize * 3 {
                return Err("JPEG ingress requires RGB8 output".into());
            }
            Ok(RgbImage {
                pixels: p.as_chunks::<3>().0.to_vec(),
                width: width as usize,
                height: height as usize,
                stride: width as usize,
            })
        }
        _ => Err("atlas supports PNG and JPEG pairs".into()),
    }
}
pub(crate) fn write(path: &Path, image: &RgbImage) -> Result<()> {
    let pixels: Vec<_> = (0..image.height)
        .flat_map(|y| image.row(y).iter())
        .map(|p| rgb::Rgb::new(p[0], p[1], p[2]))
        .collect();
    let bytes = zenpng::encode_rgb8(
        imgref::ImgRef::new(&pixels, image.width, image.height),
        None,
        &zenpng::EncodeConfig::default(),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(&bytes)?;
    Ok(())
}
pub(crate) fn preview(image: &RgbImage) -> RgbImage {
    let f = image.width.max(image.height).div_ceil(512);
    let width = image.width.div_ceil(f);
    let height = image.height.div_ceil(f);
    RgbImage {
        pixels: (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    image.pixels[(y * f).min(image.height - 1) * image.stride
                        + (x * f).min(image.width - 1)]
                })
            })
            .collect(),
        width,
        height,
        stride: width,
    }
}

impl RgbImage {
    pub fn row(&self, y: usize) -> &[[u8; 3]] {
        &self.pixels[y * self.stride..y * self.stride + self.width]
    }
    pub fn with_padding(self, padding: usize) -> Self {
        if padding == 0 {
            return self;
        }
        let stride = self.width + padding;
        let mut pixels = vec![[239u8, 17, 193]; stride * self.height];
        for y in 0..self.height {
            pixels[y * stride..y * stride + self.width].copy_from_slice(self.row(y));
        }
        Self {
            pixels,
            width: self.width,
            height: self.height,
            stride,
        }
    }
}
impl zensim::source::ImageSource for RgbImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> zensim::source::PixelFormat {
        zensim::source::PixelFormat::Srgb8Rgb
    }
    fn alpha_mode(&self) -> zensim::source::AlphaMode {
        zensim::source::AlphaMode::Opaque
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.pixels[y * self.stride..(y + 1) * self.stride])
    }
}
