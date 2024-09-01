use crate::dudx::right_cauchy_green_tensor;

//
fn add_wdwddw_from_energy_density_cauchy<Real>(
    w: &mut Real,
    dwdx: &mut [[Real; 3]; 8],
    ddwddx: &mut [[[Real; 9]; 8]; 8],
    dudx: &[[Real; 3]; 3],
    dndx: &[[Real; 3]; 8],
    wr: Real,
    dwrdc: &[Real; 6],
    ddwrddc: &[[Real; 6]; 6],
    detwei: Real,
) where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let zero = Real::zero();
    let one = Real::one();
    let two = one + one;
    let mut dcdu0 = [[[zero; 6]; 3]; 8];
    {
        let z_mat = crate::dudx::def_grad_tensor(dudx);
        for (ino, idim) in itertools::iproduct!(0..8, 0..3) {
            dcdu0[ino][idim][0] = dndx[ino][0] * z_mat[idim][0];
            dcdu0[ino][idim][1] = dndx[ino][1] * z_mat[idim][1];
            dcdu0[ino][idim][2] = dndx[ino][2] * z_mat[idim][2];
            dcdu0[ino][idim][3] = dndx[ino][0] * z_mat[idim][1] + dndx[ino][1] * z_mat[idim][0];
            dcdu0[ino][idim][4] = dndx[ino][1] * z_mat[idim][2] + dndx[ino][2] * z_mat[idim][1];
            dcdu0[ino][idim][5] = dndx[ino][2] * z_mat[idim][0] + dndx[ino][0] * z_mat[idim][2];
        }
    }
    // make ddW
    for (ino, jno) in itertools::iproduct!(0..8, 0..8) {
        for (idim, jdim) in itertools::iproduct!(0..3, 0..3) {
            let mut dtmp1 = zero;
            for (gstdim, hstdim) in itertools::iproduct!(0..6, 0..6) {
                dtmp1 += two
                    * ddwrddc[gstdim][hstdim]
                    * dcdu0[ino][idim][gstdim]
                    * dcdu0[jno][jdim][hstdim];
            }
            ddwddx[ino][jno][idim * 3 + jdim] += two * detwei * dtmp1;
        }
        {
            let mut dtmp2 = zero;
            dtmp2 += dwrdc[0] * dndx[ino][0] * dndx[jno][0];
            dtmp2 += dwrdc[1] * dndx[ino][1] * dndx[jno][1];
            dtmp2 += dwrdc[2] * dndx[ino][2] * dndx[jno][2];
            dtmp2 += dwrdc[3] * (dndx[ino][0] * dndx[jno][1] + dndx[ino][1] * dndx[jno][0]);
            dtmp2 += dwrdc[4] * (dndx[ino][1] * dndx[jno][2] + dndx[ino][2] * dndx[jno][1]);
            dtmp2 += dwrdc[5] * (dndx[ino][2] * dndx[jno][0] + dndx[ino][0] * dndx[jno][2]);
            for idim in 0..3 {
                ddwddx[ino][jno][idim * 3 + idim] += two * detwei * dtmp2;
            }
        }
    }
    // make dW
    for (ino, idim) in itertools::iproduct!(0..8, 0..3) {
        let mut dtmp1 = zero;
        for istdim in 0..6 {
            dtmp1 += dcdu0[ino][idim][istdim] * dwrdc[istdim];
        }
        dwdx[ino][idim] += two * detwei * dtmp1;
    }
    *w += wr * detwei;
}

pub fn wdwddw_compression<Real>(
    stiff_comp: Real,
    node2xyz: &[[Real; 3]; 8],
    node2disp: &[[Real; 3]; 8],
    i_gauss_degree: usize,
) -> (Real, [[Real; 3]; 8], [[[Real; 9]; 8]; 8])
where
    Real: num_traits::Float + std::ops::AddAssign + 'static,
    crate::quadrature_line::Quad<Real>: crate::quadrature_line::Quadrature<Real>,
{
    let zero = Real::zero();
    let mut w = zero;
    let mut dw = [[zero; 3]; 8];
    let mut ddw = [[[zero; 9]; 8]; 8];
    use crate::quadrature_line::Quadrature;
    let quadrature = crate::quadrature_line::Quad::<Real>::hoge(i_gauss_degree);
    let num_quadr = quadrature.len();
    for (ir1, ir2, ir3) in itertools::iproduct!(0..num_quadr, 0..num_quadr, 0..num_quadr) {
        let (dndx, detwei) = del_geo_core::hex::grad_shapefunc(node2xyz, quadrature, ir1, ir2, ir3);
        let dudx = crate::dndx::disp_grad_tensor::<8, 3, Real>(&dndx, node2disp);
        let c = right_cauchy_green_tensor::<3, Real>(&dudx);
        let (wr, dwrdc, ddwrddc) = crate::dudx::wr_dwrdc_ddwrddc_energy_density_sqr_compression(&c);
        add_wdwddw_from_energy_density_cauchy(
            &mut w,
            &mut dw,
            &mut ddw,
            &dudx,
            &dndx,
            wr,
            &dwrdc,
            &ddwrddc,
            detwei * stiff_comp,
        );
    }
    (w, dw, ddw)
}

#[test]
fn test_wdwddw_compression() {
    let node2xyz: [[f64; 3]; 8] = [
        [-1.1, -1.1, -0.9],
        [0.8, -1.0, -1.2],
        [1.3, 1.3, -1.1],
        [-1.2, 1.2, -1.3],
        [-0.9, -0.8, 1.2],
        [0.8, -1.1, 1.2],
        [1.1, 0.9, 0.9],
        [-1.3, 0.8, 1.1],
    ];
    let node2disp0 = [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, -0.1],
        [-0.1, 0.1, 0.2],
        [-0.1, -0.1, -0.3],
        [0.1, 0.1, 0.2],
        [0.3, -0.2, 0.3],
        [-0.3, 0.2, 0.1],
        [-0.2, 0.3, -0.1],
    ];
    let (w0, dw0, ddw0) = wdwddw_compression(1.0, &node2xyz, &node2disp0, 0);
    let eps = 1.0e-5;
    for (i_node, i_dim) in itertools::iproduct!(0..8, 0..3) {
        let node2disp1 = {
            let mut node2disp1 = node2disp0;
            node2disp1[i_node][i_dim] += eps;
            node2disp1
        };
        let (w1, dw1, _ddw1) = wdwddw_compression(1.0, &node2xyz, &node2disp1, 0);
        {
            let v_num = (w1 - w0) / eps;
            let v_ana = dw0[i_node][i_dim];
            assert!((v_num - v_ana).abs() < 1.0e-5);
            // dbg!((v_num - v_ana).abs());
        }
        for (j_node, j_dim) in itertools::iproduct!(0..8, 0..3) {
            let v_num = (dw1[j_node][j_dim] - dw0[j_node][j_dim]) / eps;
            let v_ana = ddw0[i_node][j_node][i_dim * 3 + j_dim];
            assert!((v_num - v_ana).abs() < 1.0e-5);
            // println!("{} {} {} {} {} {}", i_node, i_dim, j_node, j_dim, v_num, v_ana);
        }
    }
}
