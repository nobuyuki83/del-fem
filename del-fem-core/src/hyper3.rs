fn right_cauchy_green_tensor_from_disp_grad_tensor<const NDIM: usize>(
    dudx: &[[f64; NDIM]; NDIM]) -> [[f64; NDIM]; NDIM]
{
    let mut C = [[0f64; NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            C[idim][jdim] = dudx[idim][jdim] + dudx[jdim][idim];
            for kdim in 0..NDIM {
                C[idim][jdim] += dudx[kdim][idim] * dudx[kdim][jdim];
            }
        }
        C[idim][idim] += 1.0;
    }
    C
}

fn wdwddw_sqr_compression_energy_wrt_disp_grad_tensor(
    dudx: &[[f64; 3]; 3]) -> (f64, [f64; 6], [[f64; 6]; 6])
{
    // right cauchy-green tensor
    let C = right_cauchy_green_tensor_from_disp_grad_tensor::<3>(dudx);

    let (detC, Cinv) = del_geo_core::mat3_array_of_array::det_inv(&C);

    // extracting independent component of symmetric tensor
    let dWdC2 = {
        let tmp0 = (detC - 1.) * 2. * detC;
        [
            tmp0 * Cinv[0][0],
            tmp0 * Cinv[1][1],
            tmp0 * Cinv[2][2],
            tmp0 * Cinv[0][1],
            tmp0 * Cinv[1][2],
            tmp0 * Cinv[2][0],
        ]
    };
    let ddWddC2 = { // Extracting independent components in the constitutive tensor
        let istdim2ij: [[usize; 2]; 6] = [
            [0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]];
        let mut ddWddC2 = [[0f64; 6]; 6];
        for istdim in 0..6 {
            for jstdim in 0..6 {
                let idim = istdim2ij[istdim][0];
                let jdim = istdim2ij[istdim][1];
                let kdim = istdim2ij[jstdim][0];
                let ldim = istdim2ij[jstdim][1];
                let ddp3CddC = detC * (
                    4. * Cinv[idim][jdim] * Cinv[kdim][ldim]
                        - 2. * Cinv[idim][ldim] * Cinv[jdim][kdim]
                        - 2. * Cinv[idim][kdim] * Cinv[jdim][ldim]);
                let dp3CdpC_dp3CdpC = 4. * detC * detC * Cinv[idim][jdim] * Cinv[kdim][ldim];
                ddWddC2[istdim][jstdim] = (detC - 1.) * ddp3CddC + dp3CdpC_dp3CdpC;
            }
        }
        ddWddC2
    };
    return (0.5 * (detC - 1.) * (detC - 1.), dWdC2, ddWddC2);
}