#[derive(Clone)]
pub struct RigidBody {
    pub vtx2xy: Vec<f32>,
    pub pos_cg_ref: [f32; 2],
    pub pos_cg_def: [f32; 2],
    pub velo: [f32; 2],
    pub theta: f32,
    pub omega: f32,
    pub is_fix: bool,
    pub pos_cg_tmp: [f32; 2],
    pub theta_tmp: f32,
    pub mass: f32,
    pub moment_of_inertia: f32,
}

impl RigidBody {
    pub fn local2world(&self) -> [f32; 9] {
        let t0 = del_geo_core::mat3_col_major::from_translate(&[
            -self.pos_cg_ref[0],
            -self.pos_cg_ref[1],
        ]);
        let r = del_geo_core::mat3_col_major::from_rotate_z(self.theta);
        let t1 = del_geo_core::mat3_col_major::from_translate(&self.pos_cg_def);
        let rt0 = del_geo_core::mat3_col_major::mult_mat_col_major(&r, &t0);
        del_geo_core::mat3_col_major::mult_mat_col_major(&t1, &rt0)
    }

    pub fn finalize_pbd_step(&mut self, dt: f32) {
        if self.is_fix {
            self.velo = [0f32; 2];
            self.omega = 0.0;
            return;
        }
        self.velo = [
            (self.pos_cg_tmp[0] - self.pos_cg_def[0]) / dt,
            (self.pos_cg_tmp[1] - self.pos_cg_def[1]) / dt,
        ];
        self.omega = (self.theta_tmp - self.theta) / dt;
        self.pos_cg_def = self.pos_cg_tmp;
        self.theta = self.theta_tmp;
    }

    pub fn initialize_pbd_step(&mut self, dt: f32, gravity: &[f32; 2]) {
        if self.is_fix {
            self.pos_cg_tmp = self.pos_cg_def;
            self.theta_tmp = self.theta;
            return;
        }
        self.velo = [
            self.velo[0] + gravity[0] * dt,
            self.velo[1] + gravity[1] * dt,
        ];
        self.pos_cg_tmp = [
            self.pos_cg_def[0] + dt * self.velo[0],
            self.pos_cg_def[1] + dt * self.velo[1],
        ];
        self.theta_tmp = self.theta + dt * self.omega;
    }
}

pub fn resolve_contact(
    rb_a: &mut RigidBody,
    rb_b: &mut RigidBody,
    penetration: f32,
    p_a: &[f32; 2],
    p_b: &[f32; 2],
    n_b: &[f32; 2],
) -> f32 {
    use del_geo_core::vec2;
    let mut deno = 0f32;
    if !rb_a.is_fix {
        deno += 1. / rb_a.mass;
        let t0 = vec2::area_quadrilateral(&vec2::sub(p_a, &rb_a.pos_cg_def), n_b);
        deno += t0 * t0 / rb_a.moment_of_inertia;
    }
    if !rb_b.is_fix {
        deno += 1. / rb_b.mass;
        let t0 = vec2::area_quadrilateral(&vec2::sub(p_b, &rb_b.pos_cg_def), n_b);
        deno += t0 * t0 / rb_b.moment_of_inertia;
    }
    let lambda = penetration / deno; // force*dt*dt
    if !rb_a.is_fix {
        rb_a.pos_cg_tmp = [
            rb_a.pos_cg_tmp[0] + (lambda / rb_a.mass) * n_b[0],
            rb_a.pos_cg_tmp[1] + (lambda / rb_a.mass) * n_b[1],
        ];
        let t_a = vec2::area_quadrilateral(&vec2::sub(p_a, &rb_a.pos_cg_def), n_b);
        rb_a.theta_tmp += t_a * lambda / rb_a.moment_of_inertia;
    }
    if !rb_b.is_fix {
        rb_b.pos_cg_tmp = [
            rb_b.pos_cg_tmp[0] - (lambda / rb_b.mass) * n_b[0],
            rb_b.pos_cg_tmp[1] - (lambda / rb_b.mass) * n_b[1],
        ];
        let t_b = -vec2::area_quadrilateral(&vec2::sub(p_b, &rb_b.pos_cg_def), n_b);
        rb_b.theta_tmp += t_b * lambda / rb_b.moment_of_inertia;
    }
    lambda
}

/// applying PBD attachment constraint
/// # Argument
/// * `pos_a_attach` - attachment point in the local coordinate
///
/// # Return
/// * `lambda` - coefficient in the projection
pub fn attach(
    rb_a: &mut RigidBody,
    pos_a_attach: &[f32; 2],
    p_b: &mut [f32; 2],
    p_b_mass: f32,
    damp: f32,
) -> f32 {
    use del_geo_core::vec2;
    let p_a =
        del_geo_core::mat3_col_major::transform_homogeneous(&rb_a.local2world(), pos_a_attach)
            .unwrap();

    let penetration = del_geo_core::edge2::length(&p_a, p_b);
    let n_b = vec2::normalize(&vec2::sub(p_b, &p_a));
    let mut deno = 0f32;
    if !rb_a.is_fix {
        deno += 1. / rb_a.mass;
        let t0 = vec2::area_quadrilateral(&vec2::sub(&p_a, &rb_a.pos_cg_def), &n_b);
        deno += t0 * t0 / rb_a.moment_of_inertia;
    }
    {
        deno += 1. / p_b_mass;
    }
    let lambda = penetration / deno; // force*dt*dt
    if !rb_a.is_fix {
        rb_a.pos_cg_tmp = [
            rb_a.pos_cg_tmp[0] + damp * (lambda / rb_a.mass) * n_b[0],
            rb_a.pos_cg_tmp[1] + damp * (lambda / rb_a.mass) * n_b[1],
        ];
        let t_a = vec2::area_quadrilateral(&vec2::sub(&p_a, &rb_a.pos_cg_def), &n_b);
        rb_a.theta_tmp += damp * t_a * lambda / rb_a.moment_of_inertia;
    }
    {
        p_b[0] += -damp * (lambda / p_b_mass) * n_b[0];
        p_b[1] += -damp * (lambda / p_b_mass) * n_b[1];
    }
    lambda
}
