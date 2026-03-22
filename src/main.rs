use eframe::egui;
use egui::{Color32, Pos2, Sense, Stroke, StrokeKind, Vec2};

// ---------------------------------------------------------------------------
// 3-D math helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    fn dot(self, o: Vec3) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }

    fn len(self) -> f32 { (self.dot(self)).sqrt() }

    fn normalize(self) -> Vec3 {
        let l = self.len();
        if l < 1e-9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(self.x / l, self.y / l, self.z / l) }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 { Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z) }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) }
}

// Column-major 3×3 rotation matrix
#[derive(Clone, Copy)]
struct Mat3 {
    col: [Vec3; 3],
}

impl Mat3 {
    fn identity() -> Self {
        Self {
            col: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.col[0].x * v.x + self.col[1].x * v.y + self.col[2].x * v.z,
            self.col[0].y * v.x + self.col[1].y * v.y + self.col[2].y * v.z,
            self.col[0].z * v.x + self.col[1].z * v.y + self.col[2].z * v.z,
        )
    }

    fn mul_mat(self, o: Mat3) -> Mat3 {
        Mat3 {
            col: [
                self.mul_vec(o.col[0]),
                self.mul_vec(o.col[1]),
                self.mul_vec(o.col[2]),
            ],
        }
    }

    /// Rodrigues' rotation formula – rotate by `angle` radians around `axis`.
    fn rotation(axis: Vec3, angle: f32) -> Self {
        let a = axis.normalize();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;
        Mat3 {
            col: [
                Vec3::new(t * a.x * a.x + c,       t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y),
                Vec3::new(t * a.x * a.y - s * a.z, t * a.y * a.y + c,       t * a.y * a.z + s * a.x),
                Vec3::new(t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Simple perspective projection
// ---------------------------------------------------------------------------

fn project(v: Vec3, canvas_size: Vec2, fov_scale: f32) -> Pos2 {
    let z_offset = 3.5_f32;
    let z = v.z + z_offset;
    let inv_z = if z.abs() < 0.01 { 0.0 } else { fov_scale / z };
    let cx = canvas_size.x / 2.0;
    let cy = canvas_size.y / 2.0;
    Pos2::new(cx + v.x * inv_z, cy - v.y * inv_z)
}

// ---------------------------------------------------------------------------
// Maximally-distant color selection via Farthest-Point Sampling in RGB space
// ---------------------------------------------------------------------------

fn rgb_distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Pick `n` colors that are maximally separated in the unit RGB cube.
/// Uses deterministic farthest-point sampling over a dense grid.
fn farthest_point_colors(n: usize) -> Vec<[f32; 3]> {
    if n == 0 { return vec![]; }

    // Build candidate pool: 9×9×9 uniform grid (includes all corners & edge midpoints)
    let steps = 8u32;
    let mut candidates: Vec<[f32; 3]> = Vec::with_capacity((steps as usize + 1).pow(3));
    for ri in 0..=steps {
        for gi in 0..=steps {
            for bi in 0..=steps {
                candidates.push([
                    ri as f32 / steps as f32,
                    gi as f32 / steps as f32,
                    bi as f32 / steps as f32,
                ]);
            }
        }
    }

    // FPS: seed from the point farthest from the cube's centre
    let center = [0.5_f32, 0.5, 0.5];
    let mut selected: Vec<[f32; 3]> = Vec::with_capacity(n);
    let first = *candidates
        .iter()
        .max_by(|a, b| {
            rgb_distance_sq(**a, center)
                .partial_cmp(&rgb_distance_sq(**b, center))
                .unwrap()
        })
        .unwrap();
    selected.push(first);

    while selected.len() < n {
        let next = *candidates
            .iter()
            .max_by(|a, b| {
                let da = selected.iter().map(|s| rgb_distance_sq(**a, *s)).fold(f32::INFINITY, f32::min);
                let db = selected.iter().map(|s| rgb_distance_sq(**b, *s)).fold(f32::INFINITY, f32::min);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        selected.push(next);
    }

    selected
}

// ---------------------------------------------------------------------------
// Cube geometry
// ---------------------------------------------------------------------------

const CUBE_VERTICES: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct ColorCubeApp {
    rotation: Mat3,
    drag_active: bool,
    last_mouse: Option<Pos2>,
    num_colors: usize,
    colors: Vec<[f32; 3]>,
}

impl ColorCubeApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            rotation: Mat3::identity(),
            drag_active: false,
            last_mouse: None,
            num_colors: 0,
            colors: vec![],
        }
    }

    fn set_num_colors(&mut self, n: usize) {
        self.num_colors = n;
        self.colors = farthest_point_colors(n);
    }
}

impl eframe::App for ColorCubeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Color Cube");
            ui.add_space(4.0);

            // ── Controls row ────────────────────────────────────────────────
            ui.horizontal(|ui| {
                ui.label("Number of colors:");

                // Decrement button
                let dec_btn = ui.add_sized(
                    [28.0, 28.0],
                    egui::Button::new(egui::RichText::new(" - ").size(16.0).strong()),
                );
                if dec_btn.clicked() && self.num_colors > 0 {
                    self.set_num_colors(self.num_colors - 1);
                }

                // Numeral display box – framed, fixed width
                let count_str = format!("{}", self.num_colors);
                egui::Frame::new()
                    .stroke(Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color))
                    .inner_margin(egui::Margin::symmetric(8_i8, 4_i8))
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.add_sized(
                            [36.0, 22.0],
                            egui::Label::new(
                                egui::RichText::new(&count_str)
                                    .monospace()
                                    .size(18.0)
                                    .strong(),
                            ),
                        );
                    });

                // Increment button
                let inc_btn = ui.add_sized(
                    [28.0, 28.0],
                    egui::Button::new(egui::RichText::new(" + ").size(16.0).strong()),
                );
                if inc_btn.clicked() {
                    self.set_num_colors(self.num_colors + 1);
                }

                ui.add_space(16.0);
                ui.label(
                    egui::RichText::new("(drag to rotate)")
                        .weak()
                        .italics()
                        .size(12.0),
                );
            });

            ui.add_space(6.0);

            // ── RGB readout boxes: appear/disappear per selected color ───────
            if !self.colors.is_empty() {
                egui::ScrollArea::vertical()
                    .id_salt("rgb_scroll")
                    .max_height(130.0)
                    .show(ui, |ui| {
                        for (i, rgb) in self.colors.iter().enumerate() {
                            let r = (rgb[0] * 255.0).round() as u8;
                            let g = (rgb[1] * 255.0).round() as u8;
                            let b = (rgb[2] * 255.0).round() as u8;
                            let col = Color32::from_rgb(r, g, b);

                            egui::Frame::new()
                                .stroke(Stroke::new(1.0, col))
                                .inner_margin(egui::Margin::symmetric(6_i8, 3_i8))
                                .corner_radius(4.0)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        // Color swatch
                                        let (rect, _) = ui.allocate_exact_size(
                                            Vec2::new(28.0, 20.0),
                                            Sense::hover(),
                                        );
                                        ui.painter().rect_filled(rect, 3.0, col);
                                        let swatch_stroke = Stroke::new(1.0, Color32::from_white_alpha(80));
                                        ui.painter().rect_stroke(rect, 3.0, swatch_stroke, StrokeKind::Middle);

                                        ui.monospace(format!(
                                            "#{:02X}{:02X}{:02X}   R:{:3}  G:{:3}  B:{:3}",
                                            r, g, b, r, g, b
                                        ));
                                        ui.label(
                                            egui::RichText::new(format!(" #{}", i + 1))
                                                .weak()
                                                .size(11.0),
                                        );
                                    });
                                });

                            ui.add_space(2.0);
                        }
                    });

                ui.add_space(6.0);
            }

            // ── 3-D canvas ───────────────────────────────────────────────────
            let available = ui.available_size();
            let canvas_size = Vec2::new(available.x, available.y.max(300.0));
            let (response, painter) = ui.allocate_painter(canvas_size, Sense::click_and_drag());

            // Mouse drag rotation
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pos) = response.interact_pointer_pos() {
                    if let Some(last) = self.last_mouse {
                        let delta = pos - last;
                        let sensitivity = 0.008_f32;
                        let ry = Mat3::rotation(Vec3::new(0.0, 1.0, 0.0), delta.x * sensitivity);
                        let rx = Mat3::rotation(Vec3::new(1.0, 0.0, 0.0), delta.y * sensitivity);
                        self.rotation = rx.mul_mat(ry).mul_mat(self.rotation);
                    }
                    self.last_mouse = Some(pos);
                    self.drag_active = true;
                }
            } else {
                self.last_mouse = None;
                self.drag_active = false;
            }

            // Continuous auto-spin when idle; always request repaint
            if !self.drag_active {
                let spin = Mat3::rotation(Vec3::new(0.3, 1.0, 0.2).normalize(), 0.005);
                self.rotation = spin.mul_mat(self.rotation);
            }
            ctx.request_repaint();

            // Canvas background
            painter.rect_filled(
                response.rect,
                8.0,
                Color32::from_rgb(12, 12, 22),
            );

            // FOV scale – larger value ⇒ bigger cube on screen
            let fov_scale = canvas_size.x.min(canvas_size.y) * 0.55;
            let center = Vec3::new(0.5, 0.5, 0.5);

            let transform = |raw: [f32; 3]| -> Vec3 {
                let v = Vec3::new(raw[0] - center.x, raw[1] - center.y, raw[2] - center.z);
                self.rotation.mul_vec(v)
            };

            let to_screen = |raw: [f32; 3]| -> Pos2 {
                let v = transform(raw);
                let sp = project(v, canvas_size, fov_scale);
                response.rect.min + sp.to_vec2()
            };

            // ── Draw cube edges ──────────────────────────────────────────────
            for edge in &CUBE_EDGES {
                let a = CUBE_VERTICES[edge[0]];
                let b = CUBE_VERTICES[edge[1]];
                let pa = to_screen(a);
                let pb = to_screen(b);
                let cr = ((a[0] + b[0]) / 2.0 * 220.0) as u8;
                let cg = ((a[1] + b[1]) / 2.0 * 220.0) as u8;
                let cb = ((a[2] + b[2]) / 2.0 * 220.0) as u8;
                painter.line_segment([pa, pb], Stroke::new(1.5, Color32::from_rgb(cr, cg, cb)));
            }

            // ── Draw corner vertices ─────────────────────────────────────────
            for vert in &CUBE_VERTICES {
                let p = to_screen(*vert);
                let r = (vert[0] * 255.0) as u8;
                let g = (vert[1] * 255.0) as u8;
                let b = (vert[2] * 255.0) as u8;
                painter.circle_filled(p, 6.0, Color32::from_rgb(r, g, b));
                painter.circle_stroke(p, 6.0, Stroke::new(1.0, Color32::from_rgba_premultiplied(255, 255, 255, 70)));
            }

            // ── Draw selected colour spheres (depth-sorted) ──────────────────
            let mut color_points: Vec<([f32; 3], f32)> = self
                .colors
                .iter()
                .map(|rgb| {
                    let v = transform(*rgb);
                    (*rgb, v.z)
                })
                .collect();
            // back-to-front so nearer spheres paint on top
            color_points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (rgb, _z) in &color_points {
                let p = to_screen(*rgb);
                let r = (rgb[0] * 255.0).round() as u8;
                let g = (rgb[1] * 255.0).round() as u8;
                let b = (rgb[2] * 255.0).round() as u8;
                let col = Color32::from_rgb(r, g, b);
                // Glow halo
                painter.circle_filled(p, 16.0, Color32::from_rgba_premultiplied(r / 2, g / 2, b / 2, 120));
                // Main sphere
                painter.circle_filled(p, 12.0, col);
                // Specular highlight
                painter.circle_stroke(p, 12.0, Stroke::new(2.0, Color32::WHITE));
                let highlight_offset = Pos2::new(p.x - 3.5, p.y - 3.5);
                painter.circle_filled(highlight_offset, 3.0, Color32::from_rgba_premultiplied(255, 255, 255, 140));
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Color Cube")
            .with_inner_size([900.0, 750.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Color Cube",
        native_options,
        Box::new(|cc| Ok(Box::new(ColorCubeApp::new(cc)))),
    )
}
