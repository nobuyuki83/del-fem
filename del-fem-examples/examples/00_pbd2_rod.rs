use del_geo_core::vec2;
use std::time::Instant;

fn example1() -> anyhow::Result<()> {
    // constant value during simulation
    let gravity = [0., -10.];
    let dt = 0.01;
    let pnt2xy_ini = {
        let num_edge = 10;
        let num_pnt = num_edge + 1;
        let theta = std::f32::consts::PI / 6f32;
        let len_edge = 1f32 / (num_edge as f32);
        let mut pnt2xy = Vec::<f32>::with_capacity(num_pnt * 2);
        for i_pnt in 0..num_pnt {
            let x = 0.0f32 + len_edge * (i_pnt as f32) * theta.cos();
            let y = 0.8f32 - len_edge * (i_pnt as f32) * theta.sin();
            pnt2xy.push(x);
            pnt2xy.push(y);
        }
        pnt2xy
    };
    let pnt2massinv = {
        let mut pnt2massinv = vec![1f32; pnt2xy_ini.len() / 2];
        pnt2massinv[0] = 0f32;
        pnt2massinv
    };

    // visualization related stuff
    let mut canvas = del_canvas::canvas_gif::Canvas::new(
        std::path::Path::new("target/00_del_fem_canvas__pbd2__1.gif"),
        (300, 300),
        &vec![0x112F41, 0xED553B, 0xF2B134, 0x068587],
    )?;
    let transform_world2pix: [f32; 9] =
        del_geo_core::mat3_col_major::from_transform_ndc2pix((canvas.width, canvas.height));

    let mut pnt2xydef = pnt2xy_ini.clone();
    let mut pnt2xynew = pnt2xydef.clone();
    let mut pnt2velo = vec![0f32; pnt2xy_ini.len()];
    //let now = Instant::now();
    //for i_iter in 0..10000 {
    for i_step in 0..1000 {
        let num_pnt = pnt2xy_ini.len() / 2;
        for i_pnt in 0..num_pnt {
            if pnt2massinv[i_pnt] == 0f32 {
                continue;
            }
            pnt2xynew[i_pnt * 2 + 0] =
                pnt2xydef[i_pnt * 2 + 0] + dt * dt * gravity[0] + dt * pnt2velo[i_pnt * 2 + 0];
            pnt2xynew[i_pnt * 2 + 1] =
                pnt2xydef[i_pnt * 2 + 1] + dt * dt * gravity[1] + dt * pnt2velo[i_pnt * 2 + 1];
        }
        for i_seg in 0..num_pnt - 1 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let p0_def = arrayref::array_ref!(pnt2xynew, ip0 * 2, 2);
            let p1_def = arrayref::array_ref!(pnt2xynew, ip1 * 2, 2);
            let p0_ini = arrayref::array_ref!(pnt2xy_ini, ip0 * 2, 2);
            let p1_ini = arrayref::array_ref!(pnt2xy_ini, ip1 * 2, 2);
            let w0 = pnt2massinv[ip0];
            let w1 = pnt2massinv[ip1];
            let len_ini = del_geo_core::edge2::length(&p0_ini, &p1_ini);
            let (dp0, dp1) = del_fem_cpu::spring2::pbd(&p0_def, &p1_def, len_ini, w0, w1);
            pnt2xynew[ip0 * 2 + 0] += dp0[0];
            pnt2xynew[ip0 * 2 + 1] += dp0[1];
            pnt2xynew[ip1 * 2 + 0] += dp1[0];
            pnt2xynew[ip1 * 2 + 1] += dp1[1];
        }
        for i in 0..pnt2xy_ini.len() {
            pnt2velo[i] = (pnt2xynew[i] - pnt2xydef[i]) / dt;
            pnt2xydef[i] = pnt2xynew[i];
        }
        if i_step % 10 == 0 {
            canvas.clear(0);
            for p in pnt2xydef.chunks(2) {
                del_canvas::rasterize::circle2::fill::<f32, u8>(
                    &mut canvas.data,
                    canvas.width,
                    &[p[0], p[1]],
                    &transform_world2pix,
                    2.,
                    1,
                );
            }
            canvas.write();
        }
    }
    //}
    //dbg!(now.elapsed());
    Ok(())
}

fn example2() -> anyhow::Result<()> {
    // constant value during simulation
    let dt = 0.01;
    let num_edge = 10;
    let len_edge = 1f32 / num_edge as f32;
    let mut pnt2xy_def = {
        use rand::Rng;
        use rand_chacha::rand_core::SeedableRng;
        let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let num_pnt = num_edge + 1;
        let mut pnt2xy = Vec::<f32>::with_capacity(num_pnt * 2);
        for _i_pnt in 0..num_pnt {
            let x = reng.random::<f32>();
            let y = reng.random::<f32>();
            pnt2xy.push(x);
            pnt2xy.push(y);
        }
        pnt2xy
    };
    let pnt2massinv = vec![1f32; pnt2xy_def.len() / 2];

    // visualization related stuff
    let mut canvas = del_canvas::canvas_gif::Canvas::new(
        std::path::Path::new("target/00_del_fem_canvas__pbd2__2.gif"),
        (300, 300),
        &vec![0x112F41, 0xED553B, 0xF2B134, 0x068587],
    )?;
    let transform_world2pix: [f32; 9] =
        del_geo_core::mat3_col_major::from_transform_ndc2pix((canvas.width, canvas.height));

    let mut pnt2xy_new = pnt2xy_def.clone();
    let mut pnt2velo = vec![0f32; pnt2xy_def.len()];
    for i_step in 0..1000 {
        let num_pnt = pnt2xy_def.len() / 2;
        for i_pnt in 0..num_pnt {
            pnt2xy_new[i_pnt * 2 + 0] = pnt2xy_def[i_pnt * 2 + 0] + dt * pnt2velo[i_pnt * 2 + 0];
            pnt2xy_new[i_pnt * 2 + 1] = pnt2xy_def[i_pnt * 2 + 1] + dt * pnt2velo[i_pnt * 2 + 1];
        }
        // edge length constraint
        for i_seg in 0..num_pnt - 1 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let p0_def = arrayref::array_ref!(pnt2xy_new, ip0 * 2, 2);
            let p1_def = arrayref::array_ref!(pnt2xy_new, ip1 * 2, 2);
            let w0 = pnt2massinv[ip0];
            let w1 = pnt2massinv[ip1];
            let (dp0, dp1) = del_fem_cpu::spring2::pbd(&p0_def, &p1_def, len_edge, w0, w1);
            pnt2xy_new[ip0 * 2 + 0] += dp0[0];
            pnt2xy_new[ip0 * 2 + 1] += dp0[1];
            pnt2xy_new[ip1 * 2 + 0] += dp1[0];
            pnt2xy_new[ip1 * 2 + 1] += dp1[1];
        }
        // angle constraint
        for i_seg in 0..num_pnt - 2 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let ip2 = i_seg + 2;
            let p0_def = arrayref::array_ref!(pnt2xy_new, ip0 * 2, 2);
            let p1_def = arrayref::array_ref!(pnt2xy_new, ip1 * 2, 2);
            let p2_def = arrayref::array_ref!(pnt2xy_new, ip2 * 2, 2);
            let w0 = pnt2massinv[ip0];
            let w1 = pnt2massinv[ip1];
            let w2 = pnt2massinv[ip2];
            let dp = del_fem_cpu::rod2::pbd(&p0_def, &p1_def, p2_def, &[w0, w1, w2], 0f32);
            //
            let damp = 0.7;
            pnt2xy_new[ip0 * 2 + 0] += damp * dp[0][0];
            pnt2xy_new[ip0 * 2 + 1] += damp * dp[0][1];
            pnt2xy_new[ip1 * 2 + 0] += damp * dp[1][0];
            pnt2xy_new[ip1 * 2 + 1] += damp * dp[1][1];
            pnt2xy_new[ip2 * 2 + 0] += damp * dp[2][0];
            pnt2xy_new[ip2 * 2 + 1] += damp * dp[2][1];
        }
        for i_pnt in 0..pnt2xy_new.len() / 2 {
            let x = pnt2xy_new[i_pnt * 2 + 0];
            let y = pnt2xy_new[i_pnt * 2 + 1];
            let r = (x * x + y * y).sqrt();
            let r0 = 0.8;
            if r > r0 {
                let x = x / r * r0;
                let y = y / r * r0;
                pnt2xy_new[i_pnt * 2 + 0] = x;
                pnt2xy_new[i_pnt * 2 + 1] = y;
            }
        }
        for i in 0..pnt2xy_def.len() {
            pnt2velo[i] = (pnt2xy_new[i] - pnt2xy_def[i]) / dt;
            pnt2xy_def[i] = pnt2xy_new[i];
        }
        if i_step % 10 == 0 {
            canvas.clear(0);
            del_canvas::rasterize::circle2::stroke_dda::<f32, u8>(
                &mut canvas.data,
                canvas.width,
                &[0f32, 0f32],
                0.8,
                &transform_world2pix,
                2,
            );
            for p in pnt2xy_def.chunks(2) {
                del_canvas::rasterize::circle2::fill::<f32, u8>(
                    &mut canvas.data,
                    canvas.width,
                    &[p[0], p[1]],
                    &transform_world2pix,
                    2.,
                    1,
                );
            }
            canvas.write();
        }
    }
    Ok(())
}

/// the positions are constraint inside a circle
fn example3() -> anyhow::Result<()> {
    use del_geo_core::vec2::Vec2;
    // constant value during simulation
    let dt = 0.005;
    let num_edge = 10;
    let len_edge = 1f32 / num_edge as f32;
    let gravity = [0.0, 0.0];
    let radius_circle = 0.8;
    // deformed vertex positions
    let mut pnt2xy_def = {
        use rand::Rng;
        use rand_chacha::rand_core::SeedableRng;
        let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let num_pnt = num_edge + 1;
        let mut pnt2xy = Vec::<f32>::with_capacity(num_pnt * 2);
        for i_pnt in 0..num_pnt {
            let x = 0.0 + 0.001 * reng.random::<f32>();
            let y = -0.8 + 0.05 * (i_pnt as f32) + 0.001 * reng.random::<f32>();
            pnt2xy.push(x);
            pnt2xy.push(y);
        }
        pnt2xy[0] = 0.0;
        pnt2xy[1] = -0.75;
        pnt2xy
    };
    // inverse mass
    let pnt2massinv = {
        let mut pnt2massinv = vec![1f32; pnt2xy_def.len() / 2];
        pnt2massinv[0] = 0f32; // fixed condition
        pnt2massinv
    };
    // rigid body attached to the tip
    let mut rb = del_fem_cpu::pbd_rigidbody2::RigidBody {
        vtx2xy: vec![-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1],
        pos_cg_ref: [0.0, 0.0],
        pos_cg_def: [0.0, 0.0],
        velo: [0.0, 0.0],
        theta: 0f32,
        omega: 0f32,
        is_fix: false,
        pos_cg_tmp: [0.0, 0.0],
        theta_tmp: 0f32,
        mass: 0f32,
        moment_of_inertia: 0.0,
    };
    // initializing the rigid body
    {
        let rho = 1000.0;
        let area = del_msh_cpu::polyloop2::area(&rb.vtx2xy);
        let cg = del_msh_cpu::polyloop2::cog_as_face(&rb.vtx2xy);
        let moment = del_msh_cpu::polyloop2::moment_of_inertia(&rb.vtx2xy, &cg);
        rb.mass = rho * area;
        rb.moment_of_inertia = rho * moment;
        rb.pos_cg_ref = cg;
    }
    let pos_attach = [-0.0, -0.13];
    // let pos_attach = [-0.0, -0.0];
    // visualization related stuff
    let mut canvas = del_canvas::canvas_gif::Canvas::new(
        std::path::Path::new("target/00_del_fem_canvas__pbd2__3.gif"),
        (300, 300),
        &vec![0x112F41, 0xED553B, 0xF2B134, 0x068587],
    )?;
    let transform_world2pix: [f32; 9] =
        del_geo_core::mat3_col_major::from_transform_ndc2pix((canvas.width, canvas.height));
    let mut pnt2xy_new = pnt2xy_def.clone();
    let mut pnt2velo = vec![0f32; pnt2xy_def.len()];
    for i_step in 0..1000 {
        let num_pnt = pnt2xy_def.len() / 2;
        for i_pnt in 0..num_pnt {
            pnt2xy_new[i_pnt * 2 + 0] = pnt2xy_def[i_pnt * 2 + 0] + dt * pnt2velo[i_pnt * 2 + 0];
            pnt2xy_new[i_pnt * 2 + 1] = pnt2xy_def[i_pnt * 2 + 1] + dt * pnt2velo[i_pnt * 2 + 1];
        }
        rb.initialize_pbd_step(dt, &gravity);
        // initialize PBD
        // --------------------
        // edge length constraint
        for i_seg in 0..num_pnt - 1 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let (dp0, dp1) = del_fem_cpu::spring2::pbd(
                arrayref::array_ref!(pnt2xy_new, ip0 * 2, 2),
                arrayref::array_ref!(pnt2xy_new, ip1 * 2, 2),
                len_edge,
                pnt2massinv[ip0],
                pnt2massinv[ip1],
            );
            arrayref::array_mut_ref![pnt2xy_new, ip0 * 2, 2].add_in_place(&dp0);
            arrayref::array_mut_ref![pnt2xy_new, ip1 * 2, 2].add_in_place(&dp1);
        }
        // angle constraint
        for i_seg in 0..num_pnt - 2 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let ip2 = i_seg + 2;
            let dp = del_fem_cpu::rod2::pbd(
                arrayref::array_ref!(pnt2xy_new, ip0 * 2, 2),
                arrayref::array_ref!(pnt2xy_new, ip1 * 2, 2),
                arrayref::array_ref!(pnt2xy_new, ip2 * 2, 2),
                &[pnt2massinv[ip0], pnt2massinv[ip1], pnt2massinv[ip2]],
                0f32,
            );
            //
            let damp = 0.7;
            arrayref::array_mut_ref![pnt2xy_new, ip0 * 2, 2].add_in_place(&dp[0].scale(damp));
            arrayref::array_mut_ref![pnt2xy_new, ip1 * 2, 2].add_in_place(&dp[1].scale(damp));
            arrayref::array_mut_ref![pnt2xy_new, ip2 * 2, 2].add_in_place(&dp[2].scale(damp));
        }
        // rigid body attach constraint
        {
            let i_pnt = pnt2xy_new.len() / 2 - 1;
            let p0 = arrayref::array_mut_ref!(pnt2xy_new, i_pnt * 2, 2);
            let damp = 1.0;
            del_fem_cpu::pbd_rigidbody2::attach(
                &mut rb,
                &pos_attach,
                p0,
                1.0 / pnt2massinv[i_pnt],
                damp,
            );
        }
        // collision of vtx2xy against circle boundary
        for i_pnt in 0..pnt2xy_new.len() / 2 {
            let x = pnt2xy_new[i_pnt * 2 + 0];
            let y = pnt2xy_new[i_pnt * 2 + 1];
            let r = (x * x + y * y).sqrt();
            let r0 = radius_circle;
            if r > r0 {
                let x = x / r * r0;
                let y = y / r * r0;
                pnt2xy_new[i_pnt * 2 + 0] = x;
                pnt2xy_new[i_pnt * 2 + 1] = y;
            }
        }
        // collision of rb against circle boundary
        {
            let transform_local2world = rb.local2world();
            for q_local in rb.vtx2xy.chunks(2) {
                let q_local = arrayref::array_ref![q_local, 0, 2];
                let q_world = del_geo_core::mat3_col_major::transform_homogeneous(
                    &transform_local2world,
                    q_local,
                )
                .unwrap();
                if q_world.norm() < radius_circle {
                    continue;
                }
                let p_world = q_world.normalize().scale(radius_circle);
                let v_pq_world = vec2::sub(&q_world, &p_world);
                let penetration = v_pq_world.norm();
                let u_pq_world = v_pq_world.scale(1. / penetration);
                let dtheta0 = vec2::area_quadrilateral(&q_world.sub(&rb.pos_cg_tmp), &u_pq_world);
                let dh_theta = dtheta0 * dtheta0 / rb.moment_of_inertia;
                let dh_q = 1. / rb.mass;
                let lambda = penetration / (dh_theta + dh_q);
                let damp = 0.1;
                rb.pos_cg_tmp[0] -= damp * lambda * u_pq_world[0] / rb.mass;
                rb.pos_cg_tmp[1] -= damp * lambda * u_pq_world[1] / rb.mass;
                rb.theta_tmp -= damp * lambda * dtheta0 / rb.moment_of_inertia;
            }
        }
        // collision against rigid body
        {
            let damp = 1.01;
            let transform_local2world = rb.local2world();
            let transform_world2local =
                del_geo_core::mat3_col_major::try_inverse(&transform_local2world).unwrap();
            for i_pnt in 0..pnt2xy_new.len() / 2 {
                let p_world = arrayref::array_ref![pnt2xy_new, i_pnt * 2, 2];
                let p_local = del_geo_core::mat3_col_major::transform_homogeneous(
                    &transform_world2local,
                    p_world,
                )
                .unwrap();
                let is_inside = del_msh_cpu::polyloop2::is_include_a_point(&rb.vtx2xy, &p_local);
                if !is_inside {
                    continue;
                }
                // println!("inside {} {}", i_pnt, i_step);
                let q_local = del_msh_cpu::polyloop2::nearest_to_point(&rb.vtx2xy, &p_local)
                    .unwrap()
                    .1;
                let q_world = del_geo_core::mat3_col_major::transform_homogeneous(
                    &transform_local2world,
                    &q_local,
                )
                .unwrap();
                let v_pq_world = vec2::sub(&q_world, p_world);
                let penetration = v_pq_world.norm();
                let u_pq_world = v_pq_world.scale(1. / penetration);
                // how rigid body rotation change w.r.t. the unit force `u_pq_world`
                let dtheta0 = vec2::area_quadrilateral(&q_world.sub(&rb.pos_cg_tmp), &u_pq_world);
                let dh_theta = dtheta0 * dtheta0 / rb.moment_of_inertia;
                let dh_p = if pnt2massinv[i_pnt] == 0. {
                    0.
                } else {
                    pnt2massinv[i_pnt]
                };
                let dh_q = 1. / rb.mass;
                let lambda = penetration / (dh_theta + dh_q + dh_p);
                if pnt2massinv[i_pnt] != 0. {
                    pnt2xy_new[i_pnt * 2] += damp * lambda * u_pq_world[0] * pnt2massinv[i_pnt];
                    pnt2xy_new[i_pnt * 2 + 1] += damp * lambda * u_pq_world[1] * pnt2massinv[i_pnt];
                }
                rb.pos_cg_tmp[0] -= damp * lambda * u_pq_world[0] / rb.mass;
                rb.pos_cg_tmp[1] -= damp * lambda * u_pq_world[1] / rb.mass;
                rb.theta_tmp -= damp * lambda * dtheta0 / rb.moment_of_inertia;
                // dbg!(lambda, u_pq_world, rb.theta_tmp, penetration, pnt2massinv[i_pnt]);
                // println!("{} {} {}", i_step, i_pnt, penetration);
            }
        }
        // ------------
        for i in 0..pnt2xy_def.len() {
            pnt2velo[i] = (pnt2xy_new[i] - pnt2xy_def[i]) / dt;
            pnt2xy_def[i] = pnt2xy_new[i];
        }
        rb.finalize_pbd_step(dt);
        // -------------
        if i_step % 5 == 0 {
            canvas.clear(0);
            del_canvas::rasterize::circle2::stroke_dda::<f32, u8>(
                &mut canvas.data,
                canvas.width,
                &[0f32, 0f32],
                radius_circle,
                &transform_world2pix,
                2,
            );

            for p in pnt2xy_def.chunks(2) {
                del_canvas::rasterize::circle2::fill::<f32, u8>(
                    &mut canvas.data,
                    canvas.width,
                    &[p[0], p[1]],
                    &transform_world2pix,
                    2.,
                    1,
                );
            }
            {
                let obj2world = rb.local2world();
                let obj2pix = del_geo_core::mat3_col_major::mult_mat_col_major(
                    &transform_world2pix,
                    &obj2world,
                );
                del_canvas::rasterize::polygon2::stroke::<f32, u8>(
                    &mut canvas.data,
                    canvas.width,
                    &rb.vtx2xy,
                    &obj2pix,
                    0.6,
                    1,
                );
            }
            canvas.write();
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    example1()?;
    example2()?;
    example3()?;
    Ok(())
}
