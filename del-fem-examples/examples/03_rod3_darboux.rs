#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example
#![allow(unsafe_code)]
#![allow(clippy::undocumented_unsafe_blocks)]

use eframe::{egui, egui_glow, glow};

use del_geo_core::vecn::VecN;
use egui::mutex::Mutex;
use glow::HasContext;
use std::sync::Arc;

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([550.0, 600.0]),
        multisampling: 4,
        depth_buffer: 24,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Custom 3D painting in eframe using glow",
        options,
        Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
    )
}

struct RodSimulator<T>
where
    T: num_traits::Float,
{
    vtx2xyz_ini: Vec<T>,
    vtx2framex_ini: Vec<T>,
    vtx2xyz_def: Vec<T>,
    vtx2framex_def: Vec<T>,
    //
    vtx2isfix: Vec<[i32; 4]>,
    //
    w: T,
    dw: Vec<[T; 4]>,
    ddw: del_fem_cpu::sparse_square::Matrix<[T; 16]>,
    conv_ratio: T,
    //
    stiff_length: T,
    stiff_bendtwist: [T; 3],
}

impl<T> RodSimulator<T>
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    fn initialize_with_perturbation(&mut self, pos_mag: T, framex_mag: T) {
        use rand::Rng;
        use rand::SeedableRng;
        let one = T::one();
        let two = one + one;

        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        self.vtx2xyz_def = self.vtx2xyz_ini.clone();
        self.vtx2framex_def = self.vtx2framex_ini.clone();
        let num_vtx = self.vtx2xyz_ini.len() / 3;
        for i_vtx in 0..num_vtx {
            for i_dim in 0..3 {
                if self.vtx2isfix[i_vtx][i_dim] == 0 {
                    self.vtx2xyz_def[i_vtx * 3 + i_dim] = self.vtx2xyz_def[i_vtx * 3 + i_dim]
                        + (two * rng.random::<T>() - one) * pos_mag;
                }
            }
            if self.vtx2isfix[i_vtx][3] == 0 {
                let r: [T; 3] =
                    std::array::from_fn(|_v| (two * rng.random::<T>() - one) * framex_mag);
                self.vtx2framex_def[i_vtx * 3] = self.vtx2framex_def[i_vtx * 3] + r[0];
                self.vtx2framex_def[i_vtx * 3 + 1] = self.vtx2framex_def[i_vtx * 3 + 1] + r[1];
                self.vtx2framex_def[i_vtx * 3 + 2] = self.vtx2framex_def[i_vtx * 3 + 2] + r[2];
            }
        }
        del_fem_cpu::rod3_darboux::orthonormalize_framex_for_hair(
            &mut self.vtx2framex_def,
            &self.vtx2xyz_def,
        );
    }

    fn allocate_memory_for_linear_system(&mut self) {
        let zero = T::zero();
        let num_vtx = self.vtx2xyz_ini.len() / 3;
        self.w = zero;
        self.dw = vec![[zero; 4]; num_vtx];
        //
        {
            let (vtx2idx, idx2vtx) = del_msh_cpu::polyline::vtx2vtx_rods(&[0, num_vtx]);
            self.ddw =
                del_fem_cpu::sparse_square::Matrix::<[T; 16]>::from_vtx2vtx(&vtx2idx, &idx2vtx)
        };
    }

    fn update_static(&mut self, pick_info: &Option<(usize, [T; 3])>) {
        let zero = T::zero();
        let one = T::one();
        let num_vtx = self.vtx2xyz_ini.len() / 3;
        del_fem_cpu::rod3_darboux::wdwddw_hair_system(
            &mut self.w,
            &mut self.dw,
            &mut self.ddw,
            &self.vtx2xyz_ini,
            &self.vtx2xyz_def,
            self.stiff_length,
            &self.stiff_bendtwist,
            &self.vtx2framex_ini,
            &self.vtx2framex_def,
            T::zero(),
        );
        if let Some((i_vtx, pos_goal)) = pick_info {
            let i_vtx = *i_vtx;
            use del_geo_core::mat4_col_major;
            use mat4_col_major::Mat4ColMajor;
            let one = T::one();
            let two = one + one;
            let stiff = two * two * two * two;
            let kmat = mat4_col_major::from_diagonal(stiff, stiff, stiff, zero);
            self.ddw.row2val[i_vtx].add_in_place(&kmat);
            let c = del_geo_core::vec3::sub(
                arrayref::array_ref![self.vtx2xyz_def, i_vtx * 3, 3],
                pos_goal,
            );
            self.dw[i_vtx][0] = self.dw[i_vtx][0] + c[0];
            self.dw[i_vtx][1] = self.dw[i_vtx][1] + c[1];
            self.dw[i_vtx][2] = self.dw[i_vtx][2] + c[2];
        }
        // set bc flag
        for i_vtx in 0..num_vtx {
            for i_dof in 0..4 {
                if self.vtx2isfix[i_vtx][i_dof] == 0 {
                    continue;
                }
                self.dw[i_vtx][i_dof] = zero;
            }
        }
        self.ddw.set_fixed_dof::<4>(one, &self.vtx2isfix);
        //
        {
            let mut u_vec = vec![[zero; 4]; num_vtx];
            let mut p_vec = vec![[zero; 4]; num_vtx];
            let mut ap_vec = vec![[zero; 4]; num_vtx];
            let _hist = del_fem_cpu::sparse_square::conjugate_gradient0(
                &mut self.dw,
                &mut u_vec,
                &mut ap_vec,
                &mut p_vec,
                self.conv_ratio,
                1000,
                &self.ddw,
            );
            // dbg!(hist.last().unwrap());
            del_fem_cpu::rod3_darboux::update_solution_hair(
                &mut self.vtx2xyz_def,
                &mut self.vtx2framex_def,
                &u_vec,
                &self.vtx2isfix,
            );
        }
    }
}

struct MyApp {
    /// Behind an `Arc<Mutex<…>>` so we can pass it to [`egui::PaintCallback`] and paint later.
    drawer: Arc<Mutex<del_glow::drawer_elem2vtx_vtx2xyz::Drawer>>,
    // mat_modelview: [f32;16],
    mat_projection: [f32; 16],
    trackball: del_geo_core::view_rotation::Trackball<f32>,
    simulator: RodSimulator<f32>,
    gl: Option<Arc<glow::Context>>,
    pick_info: Option<(usize, [f32; 3])>,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut simulator = {
            let (vtx2xyz, vtx2framex) = {
                let (vtx2xyz, vtx2framex) =
                    del_fem_cpu::rod3_darboux::make_config_darboux_helix(30, 0.2, 0.2, 0.3);
                let m = del_geo_core::mat4_col_major::from_translate(&[-0.9, 0.0, 0.0]);
                let vtx2xyz = del_msh_cpu::vtx2xyz::transform_homogeneous(&vtx2xyz, &m);
                (vtx2xyz, vtx2framex)
            };
            let vtx2isfix = {
                let num_vtx = vtx2xyz.len() / 3;
                let mut vtx2isfix = vec![[0; 4]; num_vtx];
                vtx2isfix[0] = [1; 4];
                vtx2isfix[1] = [1, 1, 1, 0];
                vtx2isfix[num_vtx - 2] = [1, 1, 1, 0];
                vtx2isfix[num_vtx - 1] = [1; 4];
                vtx2isfix
            };
            RodSimulator {
                vtx2xyz_ini: vtx2xyz.clone(),
                vtx2xyz_def: vtx2xyz.clone(),
                vtx2framex_ini: vtx2framex.clone(),
                vtx2framex_def: vtx2framex,
                vtx2isfix,
                w: 0.,
                dw: vec![],
                ddw: del_fem_cpu::sparse_square::Matrix::<[f32; 16]>::new(),
                conv_ratio: 1.0e-5,
                stiff_length: 1.0,
                stiff_bendtwist: [1., 1., 1.],
            }
        };
        simulator.initialize_with_perturbation(0.5, 0.1);
        simulator.allocate_memory_for_linear_system();
        dbg!(simulator.dw.len());
        //
        let gl = cc
            .gl
            .as_ref()
            .expect("You need to run eframe with the glow backend");
        //let mut drawer = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
        // let mut drawer = del_glow::drawer_vtx2xyz::Drawer::new();
        let mut drawer = del_glow::drawer_elem2vtx_vtx2xyz::Drawer::new();
        drawer.compile_shader(gl);
        drawer.set_vtx2xyz(gl, &simulator.vtx2xyz_def, 3);
        {
            let line2vtx = del_msh_cpu::edge2vtx::from_polyline(simulator.vtx2framex_def.len() / 3);
            drawer.add_elem2vtx(gl, glow::LINES, &line2vtx, [1.0, 0.0, 0.0]);
        }
        Self {
            drawer: Arc::new(Mutex::new(drawer)),
            // mat_modelview: del_geo_core::mat4_col_major::from_identity(),
            trackball: del_geo_core::view_rotation::Trackball::default(),
            mat_projection: del_geo_core::mat4_col_major::from_identity(),
            simulator,
            gl: cc.gl.clone(),
            pick_info: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        {
            self.simulator.update_static(&self.pick_info);
            self.drawer.lock().set_vtx2xyz(
                &self.gl.clone().unwrap(),
                &self.simulator.vtx2xyz_def,
                3,
            );
            // dbg!(self.simulator.w);
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("push initialize button to add random perturbation to the rod");
            let res = ui.button("Initialize");
            if res.clicked() {
                self.simulator.initialize_with_perturbation(0.3, 0.1);
            }
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let (id, rect) = ui.allocate_space(ui.available_size());
                self.handle_event(ui, rect, id);
                self.custom_painting(ui, rect);
            });
        });
        ctx.request_repaint();
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.drawer.lock().destroy(gl);
        }
    }
}

impl MyApp {
    fn handle_event(&mut self, ui: &mut egui::Ui, rect: egui::Rect, id: egui::Id) {
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());
        let ctx = ui.ctx();
        let world2ndc = {
            let mat_modelview = self.trackball.mat4_col_major();
            let mat_projection = self.mat_projection;
            del_geo_core::mat4_col_major::mult_mat_col_major(&mat_projection, &mat_modelview)
        };
        if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary) && i.modifiers.alt) {
            let xy = response.drag_motion();
            let dx = 2.0 * xy.x / rect.width();
            let dy = -2.0 * xy.y / rect.height();
            self.trackball.camera_rotation(dx, dy);
            return;
        }
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                let pos_scr = pos - rect.left_top();
                let pos_ndc = [
                    2.0 * pos_scr.x / rect.width() - 1.0,
                    1.0 - 2.0 * pos_scr.y / rect.height(),
                ];
                self.pick_info = {
                    let mut dist_vtx = Vec::<(f32, usize)>::new();
                    for (i_vtx, xyz) in self.simulator.vtx2xyz_def.chunks(3).enumerate() {
                        let xyz = arrayref::array_ref![xyz, 0, 3];
                        let vtx_pos_ndc =
                            del_geo_core::mat4_col_major::transform_homogeneous(&world2ndc, xyz)
                                .unwrap();
                        let dist_ndc = del_geo_core::edge2::length(
                            &[vtx_pos_ndc[0], vtx_pos_ndc[1]],
                            &[pos_ndc[0], pos_ndc[1]],
                        );
                        if dist_ndc < 0.05 {
                            dist_vtx.push((vtx_pos_ndc[2], i_vtx));
                            // println!("{} {}", i_vtx, dist_ndc);
                        }
                    }
                    dist_vtx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    if let Some(a) = dist_vtx.last() {
                        let i_vtx_pick = a.1;
                        let pos_ini =
                            arrayref::array_ref![self.simulator.vtx2xyz_ini, i_vtx_pick * 3, 3];
                        Some((i_vtx_pick, *pos_ini))
                    } else {
                        None
                    }
                };
            }
        }
        if let Some((i_vtx_pick, _)) = self.pick_info {
            let pos_ini = arrayref::array_ref![self.simulator.vtx2xyz_ini, i_vtx_pick * 3, 3];
            let ndc_ini =
                del_geo_core::mat4_col_major::transform_homogeneous(&world2ndc, pos_ini).unwrap();
            if response.dragged() {
                let ndc2world = del_geo_core::mat4_col_major::try_inverse(&world2ndc).unwrap();
                if let Some(pos) = response.interact_pointer_pos() {
                    let scr_goal = pos - rect.left_top();
                    let ndc_goal = [
                        2.0 * scr_goal.x / rect.width() - 1.0,
                        1.0 - 2.0 * scr_goal.y / rect.height(),
                    ];
                    let ndc_goal = [ndc_goal[0], ndc_goal[1], ndc_ini[2]];
                    let pos_goal =
                        del_geo_core::mat4_col_major::transform_homogeneous(&ndc2world, &ndc_goal)
                            .unwrap();
                    self.pick_info = Some((i_vtx_pick, pos_goal));
                }
            }
            dbg!(&self.pick_info);
        }
        if response.drag_stopped() {
            self.pick_info = None;
        }
    }
    fn custom_painting(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        // Clone locals so we can move them into the paint callback:
        let mat_modelview = self.trackball.mat4_col_major();
        let mat_projection = self.mat_projection;
        let z_flip = del_geo_core::mat4_col_major::from_diagonal(1., 1., -1., 1.);
        let mat_projection_for_opengl =
            del_geo_core::mat4_col_major::mult_mat_col_major(&z_flip, &mat_projection);
        let mvp = del_geo_core::mat4_col_major::mult_mat_col_major(
            &mat_projection_for_opengl,
            &mat_modelview,
        );
        let drawer = self.drawer.clone();
        let callback = egui::PaintCallback {
            rect,
            callback: std::sync::Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                let gl = painter.gl();
                unsafe {
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    gl.enable(glow::DEPTH_TEST);
                }
                drawer
                    .lock()
                    .draw(&gl, &mvp, &del_geo_core::mat4_col_major::from_identity());
                drawer.lock().draw_points(
                    &gl,
                    &mvp,
                    &del_geo_core::mat4_col_major::from_identity(),
                );
            })),
        };
        ui.painter().add(callback);
    }
}
