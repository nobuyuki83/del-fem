use num_traits::AsPrimitive;

pub fn optimal_rotation_for_arap_spoke<T>(
    i_vtx: usize,
    adj2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    adj2weight: &[T],
    weight_scale: T,
) -> [T; 9]
where
    T: num_traits::Float + num_traits::FloatConst + std::fmt::Debug,
{
    let p0 = arrayref::array_ref!(vtx2xyz_ini, i_vtx * 3, 3);
    let p1 = arrayref::array_ref!(vtx2xyz_def, i_vtx * 3, 3);
    let mut a = [T::zero(); 9];
    for idx in 0..adj2vtx.len() {
        let j_vtx = adj2vtx[idx];
        let q0 = arrayref::array_ref!(vtx2xyz_ini, j_vtx * 3, 3);
        let q1 = arrayref::array_ref!(vtx2xyz_def, j_vtx * 3, 3);
        let pq0 = del_geo_core::vec3::sub(q0, p0);
        let pq1 = del_geo_core::vec3::sub(q1, p1);
        let w = adj2weight[idx] * weight_scale;
        del_geo_core::mat3_col_major::add_in_place_scaled_outer_product(&mut a, w, &pq1, &pq0);
        /*
        a.m11 += w * pq1[0] * pq0[0];
        a.m12 += w * pq1[0] * pq0[1];
        a.m13 += w * pq1[0] * pq0[2];
        a.m21 += w * pq1[1] * pq0[0];
        a.m22 += w * pq1[1] * pq0[1];
        a.m23 += w * pq1[1] * pq0[2];
        a.m31 += w * pq1[2] * pq0[0];
        a.m32 += w * pq1[2] * pq0[1];
        a.m33 += w * pq1[2] * pq0[2];
         */
    }
    del_geo_core::mat3_col_major::rotational_component(&a)
}

pub fn values_of_sparse_matrix_laplacian(
    tri2vtx: &[usize],
    vtx2xyz: &[f64],
    vtx2idx: &[usize],
    idx2vtx: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let num_vtx = vtx2xyz.len() / 3;
    let mut row2val = vec![0f64; num_vtx];
    let mut idx2val = vec![0f64; idx2vtx.len()];
    let mut merge_buffer = vec![0usize; 0];
    crate::laplace_tri3::merge_from_mesh(
        tri2vtx,
        vtx2xyz,
        vtx2idx,
        idx2vtx,
        &mut row2val,
        &mut idx2val,
        &mut merge_buffer,
    );
    (row2val, idx2val)
}

#[test]
fn test_optimal_rotation_for_arap() {
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::mat4_col_major::Mat4ColMajor;
    let (tri2vtx, vtx2xyz_ini) = del_msh_core::trimesh3_primitive::capsule_yup(0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let (row2idx, idx2col) =
        del_msh_core::vtx2vtx::from_uniform_mesh(tri2vtx.as_slice(), 3, num_vtx, false);
    let (_row2val, idx2val) = values_of_sparse_matrix_laplacian(
        tri2vtx.as_slice(),
        vtx2xyz_ini.as_slice(),
        &row2idx,
        &idx2col,
    );
    let mut vtx2xyz_def = vtx2xyz_ini.clone();
    let r0 = {
        let a_mat = del_geo_core::mat4_col_major::from_bryant_angles(1., 1., 3.);
        for i_vtx in 0..vtx2xyz_def.len() / 3 {
            let p0 = del_msh_core::vtx2xyz::to_vec3(&vtx2xyz_ini, i_vtx);
            let p1 = a_mat.transform_homogeneous(&p0).unwrap();
            vtx2xyz_def[i_vtx * 3 + 0] = p1[0];
            vtx2xyz_def[i_vtx * 3 + 1] = p1[1];
            vtx2xyz_def[i_vtx * 3 + 2] = p1[2];
        }
        del_geo_core::mat4_col_major::to_mat3_col_major_xyz(&a_mat)
    };
    for i_vtx in 0..vtx2xyz_ini.len() / 3 {
        let r = optimal_rotation_for_arap_spoke(
            i_vtx,
            &idx2col[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            vtx2xyz_ini.as_slice(),
            vtx2xyz_def.as_slice(),
            &idx2val[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            -1.,
        );
        dbg!(r.determinant());
        assert!((r.determinant() - 1.0).abs() < 1.0e-5);
        assert!(r.sub(&r0).norm() < 1.0e-5);
    }
}

fn energy_par_vtx_arap_spoke<T>(
    i_vtx: usize,
    adj2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    adj2weight: &[T],
    weight_scale: T,
    rot_mat: &[T; 9],
) -> T
where
    T: num_traits::Float + Copy + std::ops::AddAssign,
{
    use del_geo_core::vec3::Vec3;
    let p0 = arrayref::array_ref!(vtx2xyz_ini, i_vtx * 3, 3);
    let p1 = arrayref::array_ref!(vtx2xyz_def, i_vtx * 3, 3);
    let mut w = T::zero();
    for idx in 0..adj2vtx.len() {
        let j_vtx = adj2vtx[idx];
        let q0 = arrayref::array_ref!(vtx2xyz_ini, j_vtx * 3, 3);
        let q1 = arrayref::array_ref!(vtx2xyz_def, j_vtx * 3, 3);
        let pq0 = del_geo_core::vec3::sub(q0, p0);
        let pq1 = del_geo_core::vec3::sub(q1, p1);
        let pq0 = del_geo_core::mat3_col_major::mult_vec(rot_mat, &pq0);
        let diff = pq1.sub(&pq0).squared_norm();
        w += adj2weight[idx] * weight_scale * diff;
    }
    w
}

pub fn energy_arap_spoke<T>(
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    weight_scale: T,
    idx2val: &[T],
    vtx2rot: &[T],
) -> T
where
    T: num_traits::Float + Copy + std::ops::AddAssign,
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    let mut tot_w = T::zero();
    for i_vtx in 0..num_vtx {
        let r0 = arrayref::array_ref![&vtx2rot, i_vtx * 9, 9];
        let i_w = energy_par_vtx_arap_spoke(
            i_vtx,
            &idx2col[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            vtx2xyz_ini,
            vtx2xyz_def,
            &idx2val[row2idx[i_vtx]..row2idx[i_vtx + 1]],
            weight_scale,
            r0,
        );
        tot_w += i_w;
    }
    tot_w
}

#[test]
fn test_energy_arap_spoke() {
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::vec3::Vec3;
    let (tri2vtx, vtx2xyz_ini) =
        del_msh_core::trimesh3_primitive::capsule_yup::<f64>(0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let (vtx2idx, idx2vtx) =
        del_msh_core::vtx2vtx::from_uniform_mesh(tri2vtx.as_slice(), 3, num_vtx, false);
    let (_row2val, idx2val) = values_of_sparse_matrix_laplacian(
        tri2vtx.as_slice(),
        vtx2xyz_ini.as_slice(),
        &vtx2idx,
        &idx2vtx,
    );
    let vtx2xyz_def = {
        let mut vtx2xyz_def = vec![0f64; vtx2xyz_ini.len()];
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    };
    del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
        "../target/hoge.obj",
        tri2vtx.as_slice(),
        vtx2xyz_def.as_slice(),
        3,
    )
    .unwrap();
    for i_vtx in 0..num_vtx {
        let r0 = optimal_rotation_for_arap_spoke(
            i_vtx,
            &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
            vtx2xyz_ini.as_slice(),
            vtx2xyz_def.as_slice(),
            &idx2val,
            -1.0,
        );
        let e0 = energy_par_vtx_arap_spoke(
            i_vtx,
            &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
            vtx2xyz_ini.as_slice(),
            &vtx2xyz_def,
            &idx2val,
            -1.0,
            &r0,
        );
        let eps = 0.001;
        for i in 0..3 {
            let mut rot = [0f64; 3];
            rot[i] = eps;
            let r1 = r0.mult_mat_col_major(&del_geo_core::mat3_col_major::from_bryant_angles(
                rot[0], rot[1], rot[2],
            ));
            let e1 = energy_par_vtx_arap_spoke(
                i_vtx,
                &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
                vtx2xyz_ini.as_slice(),
                &vtx2xyz_def,
                &idx2val,
                -1.0,
                &r1,
            );
            assert!(e1 - e0 > 0.);
        }
    }
    let vtx2rot = {
        let mut vtx2rot = vec![0f64; num_vtx * 9];
        for i_vtx in 0..num_vtx {
            let r0 = optimal_rotation_for_arap_spoke(
                i_vtx,
                &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]],
                vtx2xyz_ini.as_slice(),
                &vtx2xyz_def,
                &idx2val,
                -1.0,
            );
            // transpose to change column-major to row-major
            r0.transpose()
                .iter()
                .enumerate()
                .for_each(|(i, &v)| vtx2rot[i_vtx * 9 + i] = v);
        }
        vtx2rot
    };
    let tot_w0 = energy_arap_spoke(
        vtx2xyz_ini.as_slice(),
        vtx2xyz_def.as_slice(),
        &vtx2idx,
        &idx2vtx,
        -1.,
        &idx2val,
        &vtx2rot,
    );
    let eps = 1.0e-5;
    for i_vtx in 0..num_vtx {
        let res = {
            let mut res = [0f64; 3];
            let p0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini.as_slice(), i_vtx);
            let p1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def.as_slice(), i_vtx);
            let r_i = arrayref::array_ref![&vtx2rot, i_vtx * 9, 9];
            for jdx in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
                let j_vtx = idx2vtx[jdx];
                let q0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini.as_slice(), j_vtx);
                let q1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def.as_slice(), j_vtx);
                let r_j = arrayref::array_ref![&vtx2rot, j_vtx * 9, 9];
                let weight = -idx2val[jdx];
                let rm = r_i.add(&r_j).scale(0.5);
                let d1 = q1.sub(p1);
                let d0 = q0.sub(p0);
                let diff = d1.sub(&rm.mult_vec(&d0)).scale(-4. * weight);
                res.add_in_place(&diff);
            }
            res
        };
        for i_dim in 0..3 {
            let vtx2xyz_ptb = {
                let mut vtx2xyz_ptb = vtx2xyz_def.clone();
                vtx2xyz_ptb[i_vtx * 3 + i_dim] += eps;
                vtx2xyz_ptb
            };
            let tot_w1 = energy_arap_spoke(
                vtx2xyz_ini.as_slice(),
                vtx2xyz_ptb.as_slice(),
                &vtx2idx,
                &idx2vtx,
                -1.,
                &idx2val,
                &vtx2rot,
            );
            let dwdp = (tot_w1 - tot_w0) / eps;
            // dbg!(res[i_dim], dwdp);
            assert!((res[i_dim] - dwdp).abs() < dwdp.abs() * 0.001 + 0.0002);
        }
    }
}

pub fn optimal_rotations_mesh_vertx_for_arap_spoke_rim<T>(
    vtx2rot: &mut [T],
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
) where
    T: num_traits::Float + num_traits::FloatConst + 'static + Copy + std::fmt::Debug,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    vtx2rot.fill(T::zero());
    for nodes in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (nodes[0], nodes[1], nodes[2]);
        let cots = del_geo_core::tri3::cot(
            &vtx2xyz_ini[i0 * 3..i0 * 3 + 3].try_into().unwrap(),
            &vtx2xyz_ini[i1 * 3..i1 * 3 + 3].try_into().unwrap(),
            &vtx2xyz_ini[i2 * 3..i2 * 3 + 3].try_into().unwrap(),
        );
        let p0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i0);
        let p1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i1);
        let p2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i2);
        let q0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i0);
        let q1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i1);
        let q2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i2);
        // nalgebra matrix for R^T to make 'vtx2rot' row-major order
        let rt = del_geo_core::mat3_col_major::add_three(
            &del_geo_core::mat3_col_major::from_scaled_outer_product(
                cots[0],
                &p1.sub(p2),
                &q1.sub(q2),
            ),
            &del_geo_core::mat3_col_major::from_scaled_outer_product(
                cots[1],
                &p2.sub(p0),
                &q2.sub(q0),
            ),
            &del_geo_core::mat3_col_major::from_scaled_outer_product(
                cots[2],
                &p0.sub(p1),
                &q0.sub(q1),
            ),
        );
        vtx2rot[i0 * 9..i0 * 9 + 9]
            .iter_mut()
            .zip(rt.iter())
            .for_each(|(v, &w)| *v = *v + w);
        vtx2rot[i1 * 9..i1 * 9 + 9]
            .iter_mut()
            .zip(rt.iter())
            .for_each(|(v, &w)| *v = *v + w);
        vtx2rot[i2 * 9..i2 * 9 + 9]
            .iter_mut()
            .zip(rt.iter())
            .for_each(|(v, &w)| *v = *v + w);
    }
    for i_vtx in 0..num_vtx {
        let rt = arrayref::array_ref![&vtx2rot, i_vtx * 9, 9];
        let rt = del_geo_core::mat3_col_major::rotational_component(&rt);
        vtx2rot[i_vtx * 9..i_vtx * 9 + 9]
            .iter_mut()
            .zip(rt.iter())
            .for_each(|(v, &w)| *v = *v + w);
    }
}

struct CornerVertices<'a, T> {
    p0: &'a [T; 3],
    p1: &'a [T; 3],
    p2: &'a [T; 3],
}

fn wdw_arap_spoke_rim<T>(
    s: CornerVertices<T>,
    e: CornerVertices<T>,
    rot0: &[T; 9],
    rot1: &[T; 9],
    rot2: &[T; 9],
) -> (T, [[T; 3]; 3])
where
    T: nalgebra::RealField + Copy + std::ops::AddAssign + num_traits::Float + 'static,
    f64: AsPrimitive<T>,
{
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::vec3::Vec3;
    let cots = del_geo_core::tri3::cot(s.p0, s.p1, s.p2);
    let mut w = T::zero();
    {
        let coeff: T = (0.25f64 / 3.0f64).as_();
        let d12_0 = e.p2.sub(e.p1).sub(&rot0.mult_vec(&s.p2.sub(s.p1)));
        let d12_1 = e.p2.sub(e.p1).sub(&rot1.mult_vec(&s.p2.sub(s.p1)));
        let d12_2 = e.p2.sub(e.p1).sub(&rot2.mult_vec(&s.p2.sub(s.p1)));
        w += coeff * cots[0] * (d12_0.squared_norm() + d12_2.squared_norm() + d12_1.squared_norm());
        //
        let d20_0 = e.p0.sub(e.p2).sub(&rot0.mult_vec(&s.p0.sub(s.p2)));
        let d20_1 = e.p0.sub(e.p2).sub(&rot1.mult_vec(&s.p0.sub(s.p2)));
        let d20_2 = e.p0.sub(e.p2).sub(&rot2.mult_vec(&s.p0.sub(s.p2)));
        w += coeff * cots[1] * (d20_0.squared_norm() + d20_1.squared_norm() + d20_2.squared_norm());
        //
        let d01_0 = e.p1.sub(e.p0).sub(&rot0.mult_vec(&s.p1.sub(s.p0)));
        let d01_1 = e.p1.sub(e.p0).sub(&rot1.mult_vec(&s.p1.sub(s.p0)));
        let d01_2 = e.p1.sub(e.p0).sub(&rot2.mult_vec(&s.p1.sub(s.p0)));
        w += coeff * cots[2] * (d01_0.squared_norm() + d01_1.squared_norm() + d01_2.squared_norm());
    }
    let mut dw = [[T::zero(); 3]; 3];
    {
        let rot =
            del_geo_core::mat3_col_major::add_three(rot0, rot1, rot2).scale(T::one() / 3.0.as_());
        let d12 = e.p2.sub(e.p1).sub(&rot.mult_vec(&s.p2.sub(s.p1)));
        let d20 = e.p0.sub(e.p2).sub(&rot.mult_vec(&s.p0.sub(s.p2)));
        let d01 = e.p1.sub(e.p0).sub(&rot.mult_vec(&s.p1.sub(s.p0)));
        let coeff: T = 0.5f64.as_();
        dw[0].add_in_place(&d20.scale(coeff * cots[1]).sub(&d01.scale(coeff * cots[2])));
        dw[1].add_in_place(&d01.scale(coeff * cots[2]).sub(&d12.scale(coeff * cots[0])));
        dw[2].add_in_place(&d12.scale(coeff * cots[0]).sub(&d20.scale(coeff * cots[1])));
    }
    (w, dw)
}

#[test]
fn test_wdw_arap_spoke_rim() {
    let p0 = [0f64, 0., 0.];
    let p1 = [1., 2., 3.];
    let p2 = [2., 1., 1.];
    let q0 = [3., 2., 1.];
    let q1 = [3., 0., 4.];
    let q2 = [5., 2., 0.];
    let rot0 = del_geo_core::mat3_col_major::from_bryant_angles(1., 2., 3.);
    let rot1 = del_geo_core::mat3_col_major::from_bryant_angles(2., 3., 1.);
    let rot2 = del_geo_core::mat3_col_major::from_bryant_angles(3., 1., 2.);
    let eps = 1.0e-4;
    let (w0, dw0) = wdw_arap_spoke_rim(
        CornerVertices {
            p0: &p0,
            p1: &p1,
            p2: &p2,
        },
        CornerVertices {
            p0: &q0,
            p1: &q1,
            p2: &q2,
        },
        &rot0,
        &rot1,
        &rot2,
    );
    for i_dim in 0..3 {
        let mut q0a = q0.clone();
        q0a[i_dim] += eps;
        let (w1_0, _dw1_0) = wdw_arap_spoke_rim(
            CornerVertices {
                p0: &p0,
                p1: &p1,
                p2: &p2,
            },
            CornerVertices {
                p0: &q0a,
                p1: &q1,
                p2: &q2,
            },
            &rot0,
            &rot1,
            &rot2,
        );
        assert!(((w1_0 - w0) / eps - dw0[0][i_dim]).abs() < dw0[0][i_dim].abs() * 0.001 + 0.00001);
        //
        let mut q1a = q1.clone();
        q1a[i_dim] += eps;
        let (w1_1, _dw1_1) = wdw_arap_spoke_rim(
            CornerVertices {
                p0: &p0,
                p1: &p1,
                p2: &p2,
            },
            CornerVertices {
                p0: &q0,
                p1: &q1a,
                p2: &q2,
            },
            &rot0,
            &rot1,
            &rot2,
        );
        assert!(((w1_1 - w0) / eps - dw0[1][i_dim]).abs() < dw0[1][i_dim].abs() * 0.001 + 0.00001);
        //
        let mut q2a = q2.clone();
        q2a[i_dim] += eps;
        let (w1_2, _dw1_2) = wdw_arap_spoke_rim(
            CornerVertices {
                p0: &p0,
                p1: &p1,
                p2: &p2,
            },
            CornerVertices {
                p0: &q0,
                p1: &q1,
                p2: &q2a,
            },
            &rot0,
            &rot1,
            &rot2,
        );
        assert!(((w1_2 - w0) / eps - dw0[2][i_dim]).abs() < dw0[2][i_dim].abs() * 0.001 + 0.00001,);
    }
}

pub fn energy_arap_spoke_rim<T>(
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    vtx2rot: &[T],
) -> T
where
    T: nalgebra::RealField + Copy + std::ops::AddAssign + num_traits::Float + 'static,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    let mut tot_w = T::zero();
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i0);
        let p1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i1);
        let p2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i2);
        let q0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i0);
        let q1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i1);
        let q2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i2);
        let (w, _) = wdw_arap_spoke_rim(
            CornerVertices { p0, p1, p2 },
            CornerVertices {
                p0: q0,
                p1: q1,
                p2: q2,
            },
            arrayref::array_ref![&vtx2rot, i0 * 9, 9],
            arrayref::array_ref![&vtx2rot, i1 * 9, 9],
            arrayref::array_ref![&vtx2rot, i2 * 9, 9],
        );
        tot_w += w;
    }
    tot_w
}

#[test]
fn test_energy_arap_spoke_rim() {
    let (tri2vtx, vtx2xyz_ini) =
        del_msh_core::trimesh3_primitive::capsule_yup::<f64>(0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let vtx2xyz_def = {
        let mut vtx2xyz_def = vec![0f64; vtx2xyz_ini.len()];
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    };
    let _ = del_msh_core::io_obj::save_tri2vtx_vtx2xyz(
        "target/hoge.obj",
        tri2vtx.as_slice(),
        vtx2xyz_def.as_slice(),
        3,
    );
    let mut vtx2rot = vec![0f64; num_vtx * 9];
    optimal_rotations_mesh_vertx_for_arap_spoke_rim(
        &mut vtx2rot,
        tri2vtx.as_slice(),
        vtx2xyz_ini.as_slice(),
        &vtx2xyz_def,
    );
}

pub fn residual_arap_spoke_rim<T>(
    vtx2res: &mut [T],
    tri2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    vtx2rot: &[T],
) where
    T: num_traits::Float + 'static + nalgebra::RealField + Copy,
    f64: num_traits::AsPrimitive<T>,
{
    let num_vtx = vtx2xyz_ini.len() / 3;
    assert_eq!(vtx2rot.len(), num_vtx * 9);
    assert_eq!(vtx2res.len(), num_vtx * 3);
    vtx2res.fill(T::zero());
    for nodes in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (nodes[0], nodes[1], nodes[2]);
        let p0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i0);
        let p1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i1);
        let p2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_ini, i2);
        let q0 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i0);
        let q1 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i1);
        let q2 = del_msh_core::vtx2xyz::to_vec3(vtx2xyz_def, i2);
        let (_, dw) = wdw_arap_spoke_rim(
            CornerVertices { p0, p1, p2 },
            CornerVertices {
                p0: q0,
                p1: q1,
                p2: q2,
            },
            arrayref::array_ref![&vtx2rot, i0 * 9, 9],
            arrayref::array_ref![&vtx2rot, i1 * 9, 9],
            arrayref::array_ref![&vtx2rot, i2 * 9, 9],
        );
        vtx2res[i0 * 3..i0 * 3 + 3]
            .iter_mut()
            .zip(dw[0].iter())
            .for_each(|(x, y)| *x -= *y);
        vtx2res[i1 * 3..i1 * 3 + 3]
            .iter_mut()
            .zip(dw[1].iter())
            .for_each(|(x, y)| *x -= *y);
        vtx2res[i2 * 3..i2 * 3 + 3]
            .iter_mut()
            .zip(dw[2].iter())
            .for_each(|(x, y)| *x -= *y);
    }
}

#[cfg(test)]
mod tests {
    fn mydef(vtx2xyz_ini: &[f64]) -> Vec<f64> {
        let num_vtx = vtx2xyz_ini.len() / 3;
        let mut vtx2xyz_def = vec![0f64; vtx2xyz_ini.len()];
        for i_vtx in 0..num_vtx {
            let x0 = vtx2xyz_ini[i_vtx * 3 + 0];
            let y0 = vtx2xyz_ini[i_vtx * 3 + 1];
            let z0 = vtx2xyz_ini[i_vtx * 3 + 2];
            let x1 = x0 + 0.1 * (3.0 * y0).sin() - 0.1 * (5.0 * z0).cos();
            let y1 = y0 + 0.2 * (4.0 * x0).sin() + 0.2 * (4.0 * z0).cos();
            let z1 = z0 - 0.1 * (5.0 * x0).sin() + 0.1 * (3.0 * y0).cos();
            vtx2xyz_def[i_vtx * 3 + 0] = x1;
            vtx2xyz_def[i_vtx * 3 + 1] = y1;
            vtx2xyz_def[i_vtx * 3 + 2] = z1;
        }
        vtx2xyz_def
    }

    #[test]
    fn test_energy_arap_spoke_rim_resolution() {
        let (tri2vtx0, vtx2xyz0_ini) =
            del_msh_core::trimesh3_primitive::capsule_yup::<f64>(0.2, 1.6, 24, 4, 24);
        let vtx2rot0 = {
            let mut vtx2rot0 = vec![0f64; vtx2xyz0_ini.len() * 3];
            for i in 0..vtx2xyz0_ini.len() / 3 {
                vtx2rot0[i * 9 + 0] = 1.0;
                vtx2rot0[i * 9 + 4] = 1.0;
                vtx2rot0[i * 9 + 8] = 1.0;
            }
            vtx2rot0
        };
        let vtx2xyz0_def = mydef(vtx2xyz0_ini.as_slice());
        let w0 = crate::arap::energy_arap_spoke_rim(
            tri2vtx0.as_slice(),
            vtx2xyz0_ini.as_slice(),
            &vtx2xyz0_def,
            &vtx2rot0,
        );
        //
        let (tri2vtx1, vtx2xyz1_ini) =
            del_msh_core::trimesh3_primitive::capsule_yup::<f64>(0.2, 1.6, 48, 8, 48);
        let vtx2rot1 = {
            let mut vtx2rot1 = vec![0f64; vtx2xyz1_ini.len() * 3];
            for i in 0..vtx2xyz1_ini.len() / 3 {
                vtx2rot1[i * 9 + 0] = 1.0;
                vtx2rot1[i * 9 + 4] = 1.0;
                vtx2rot1[i * 9 + 8] = 1.0;
            }
            vtx2rot1
        };
        let vtx2xyz1_def = mydef(vtx2xyz1_ini.as_slice());
        let w1 = crate::arap::energy_arap_spoke_rim(
            tri2vtx1.as_slice(),
            vtx2xyz1_ini.as_slice(),
            &vtx2xyz1_def,
            &vtx2rot1,
        );
        //
        let (tri2vtx2, vtx2xyz2_ini) =
            del_msh_core::trimesh3_primitive::capsule_yup::<f64>(0.2, 1.6, 96, 16, 96);
        let vtx2rot2 = {
            let mut vtx2rot2 = vec![0f64; vtx2xyz2_ini.len() * 3];
            for i in 0..vtx2xyz2_ini.len() / 3 {
                vtx2rot2[i * 9 + 0] = 1.0;
                vtx2rot2[i * 9 + 4] = 1.0;
                vtx2rot2[i * 9 + 8] = 1.0;
            }
            vtx2rot2
        };
        let vtx2xyz2_def = mydef(vtx2xyz2_ini.as_slice());
        let w2 = crate::arap::energy_arap_spoke_rim(
            tri2vtx2.as_slice(),
            vtx2xyz2_ini.as_slice(),
            vtx2xyz2_def.as_slice(),
            &vtx2rot2,
        );
        assert!((w0 - w1).abs() < w1 * 0.01);
        assert!((w1 - w2).abs() < w2 * 0.004);
    }
}
