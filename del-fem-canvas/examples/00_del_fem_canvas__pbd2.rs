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

    let mut pnt2xy_def = pnt2xy_ini.clone();
    let mut pnt2xy_new = pnt2xy_def.clone();
    let mut pnt2velo = vec![0f32; pnt2xy_ini.len()];
    for i_step in 0..1000 {
        let num_pnt = pnt2xy_ini.len() / 2;
        for i_pnt in 0..num_pnt {
            if pnt2massinv[i_pnt] == 0f32 {
                continue;
            }
            pnt2xy_new[i_pnt * 2 + 0] =
                pnt2xy_def[i_pnt * 2 + 0] + dt * dt * gravity[0] + dt * pnt2velo[i_pnt * 2 + 0];
            pnt2xy_new[i_pnt * 2 + 1] =
                pnt2xy_def[i_pnt * 2 + 1] + dt * dt * gravity[1] + dt * pnt2velo[i_pnt * 2 + 1];
        }
        for i_seg in 0..num_pnt - 1 {
            let ip0 = i_seg;
            let ip1 = i_seg + 1;
            let p0_def = arrayref::array_ref!(pnt2xy_new, ip0 * 2, 2);
            let p1_def = arrayref::array_ref!(pnt2xy_new, ip1 * 2, 2);
            let p0_ini = arrayref::array_ref!(pnt2xy_ini, ip0 * 2, 2);
            let p1_ini = arrayref::array_ref!(pnt2xy_ini, ip1 * 2, 2);
            let w0 = pnt2massinv[ip0];
            let w1 = pnt2massinv[ip1];
            let len_ini = del_geo_core::edge2::length(&p0_ini, &p1_ini);
            let (dp0, dp1) = del_fem_core::spring2::pbd(&p0_def, &p1_def, len_ini, w0, w1);
            pnt2xy_new[ip0 * 2 + 0] += dp0[0];
            pnt2xy_new[ip0 * 2 + 1] += dp0[1];
            pnt2xy_new[ip1 * 2 + 0] += dp1[0];
            pnt2xy_new[ip1 * 2 + 1] += dp1[1];
        }
        for i in 0..pnt2xy_ini.len() {
            pnt2velo[i] = (pnt2xy_new[i] - pnt2xy_def[i]) / dt;
            pnt2xy_def[i] = pnt2xy_new[i];
        }
        if i_step % 10 == 0 {
            canvas.clear(0);
            for p in pnt2xy_def.chunks(2) {
                del_canvas::rasterize::circle::fill::<f32, u8>(
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
            let (dp0, dp1) = del_fem_core::spring2::pbd(&p0_def, &p1_def, len_edge, w0, w1);
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
            let dp = del_fem_core::rod2::pbd(&p0_def, &p1_def, p2_def, &[w0, w1, w2], 0f32);
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
            del_canvas::rasterize::circle::stroke_dda::<f32, u8>(
                &mut canvas.data,
                canvas.width,
                &[0f32, 0f32],
                0.8,
                &transform_world2pix,
                2,
            );
            for p in pnt2xy_def.chunks(2) {
                del_canvas::rasterize::circle::fill::<f32, u8>(
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
    let mut rb = del_fem_core::pbd_rigidbody2::RigidBody {
        vtx2xy: vec![-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1],
        pos_cg_ref: [0.0, 0.0],
        pos_cg_def: [0.0, 0.0],
        velo: [0.0, 0.0],
        theta: 0f32,
        omega: 0f32,
        is_fix: false,
        pos_tmp: [0.0, 0.0],
        theta_tmp: 0f32,
        mass: 0f32,
        moment_of_inertia: 0.0,
    };
    // initializing the rigid body
    {
        let rho = 1000.0;
        let area = del_msh_core::polyloop2::area(&rb.vtx2xy);
        let cg = del_msh_core::polyloop2::cog_as_face(&rb.vtx2xy);
        let moment = del_msh_core::polyloop2::moment_of_inertia(&rb.vtx2xy, &cg);
        rb.mass = rho * area;
        rb.moment_of_inertia = rho * moment;
        rb.pos_cg_ref = cg;
    }
    let pos_attach = [-0.0, -0.13];
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
            let (dp0, dp1) = del_fem_core::spring2::pbd(
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
            let dp = del_fem_core::rod2::pbd(
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
            del_fem_core::pbd_rigidbody2::attach(
                &mut rb,
                &pos_attach,
                p0,
                1.0 / pnt2massinv[i_pnt],
                damp,
            );
        }
        // collision against circle boundary
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
        // collision against rigid body
        {
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
                let vtx2xy = &rb.vtx2xy;
                let is_inside = del_msh_core::polyloop2::is_inside(vtx2xy, &p_local);
                if is_inside {
                    dbg!("inside {} {}", i_pnt, i_step);
                }
            }
        }
        // ------------
        for i in 0..pnt2xy_def.len() {
            pnt2velo[i] = (pnt2xy_new[i] - pnt2xy_def[i]) / dt;
            pnt2xy_def[i] = pnt2xy_new[i];
        }
        rb.finalize_pbd_step(dt);
        // -------------
        if i_step % 10 == 0 {
            canvas.clear(0);
            del_canvas::rasterize::circle::stroke_dda::<f32, u8>(
                &mut canvas.data,
                canvas.width,
                &[0f32, 0f32],
                0.8,
                &transform_world2pix,
                2,
            );
            for p in pnt2xy_def.chunks(2) {
                del_canvas::rasterize::circle::fill::<f32, u8>(
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
                del_canvas::rasterize::polygon::stroke::<f32, u8>(
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
