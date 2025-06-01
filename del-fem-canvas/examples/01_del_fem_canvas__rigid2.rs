fn step_time(rbs: &mut [del_fem_core::pbd_rigidbody2::RigidBody], dt: f32, gravity: &[f32; 2]) {
    use del_geo_core::mat3_col_major;
    rbs.iter_mut()
        .for_each(|rb| rb.initialize_pbd_step(dt, gravity));
    for irb in 0..rbs.len() {
        let lcli2world = rbs[irb].local2world();
        for jrb in 0..rbs.len() {
            if irb == jrb {
                continue;
            }
            let lclj2world = rbs[jrb].local2world();
            let lcli2lclj = mat3_col_major::mult_mat_col_major(
                &mat3_col_major::try_inverse(&lclj2world).unwrap(),
                &lcli2world,
            );
            for ivtx in 0..rbs[irb].vtx2xy.len() / 2 {
                let xyi = arrayref::array_ref!(rbs[irb].vtx2xy, ivtx * 2, 2);
                let xyi_lclj = mat3_col_major::transform_homogeneous(&lcli2lclj, xyi).unwrap();
                let (sdf, nrm) = del_msh_cpu::polyloop2::wdw_sdf(&rbs[jrb].vtx2xy, &xyi_lclj);
                if sdf > 0.0 {
                    continue;
                }
                // dbg!(irb, jrb, xyi, xyi_lclj, sdf, nrm);
                let xyi_world = mat3_col_major::transform_homogeneous(&lcli2world, &xyi).unwrap();
                let nrmj = mat3_col_major::transform_direction(&lclj2world, &nrm);
                let mut rbi = rbs[irb].clone();
                let mut rbj = rbs[jrb].clone();
                let _lambda = del_fem_core::pbd_rigidbody2::resolve_contact(
                    &mut rbi, &mut rbj, -sdf, &xyi_world, &xyi_world, &nrmj,
                );
                rbs[irb] = rbi;
                rbs[jrb] = rbj;
            }
        }
    }

    // finalize position, set velocity
    rbs.iter_mut().for_each(|rb| rb.finalize_pbd_step(dt));
}

/*
// find & resolve contact
std::vector<CContactInfo2> aContact;
for(unsigned int irbA=0;irbA<aRS.size();++irbA) { // loop over rigid bodies
for(unsigned int ip=0;ip<aRS[irbA].shape.size();++ip) {
for(unsigned int irbB=0;irbB<aRS.size();++irbB) { // loop over rigid bodies
if( irbA == irbB ){ continue; }
const CRigidState2& rbA = aRS[irbA];
const CMat3d& matAffineA = rgd_v2m3::Mat3_Affine(rbA.posl,rbA.theta_tmp,rbA.posg_tmp);
const CVec2d& pA = rbA.shape[ip].Mat3Vec2_AffineProjection(matAffineA.p_);
const CRigidState2& rbB = aRS[irbB];
const CMat3d& matAffineB = rgd_v2m3::Mat3_Affine(rbB.posl,rbB.theta_tmp,rbB.posg_tmp); // j-Rigidbody's affine matrix
const CVec2d& pAonB = rbA.shape[ip].Mat3Vec2_AffineProjection((matAffineB.Inverse() * matAffineA).p_); // Pi in j-Rigidbody's coordinate
unsigned int ieB;
double reB;
CVec2d NrmB;
const bool is_inside = rgd_v2m3::Nearest_Polygon2(ieB, reB, NrmB, rbB.shape, pAonB);
if( !is_inside ){ continue; } // not penetrating
//
CVec2d PB = (1-reB)*rbB.shape[(ieB+0)%rbB.shape.size()] + reB*rbB.shape[(ieB+1)%rbB.shape.size()];
const double penetration = (PB-pAonB).dot(NrmB);
const CVec2d pB = PB.Mat3Vec2_AffineProjection(matAffineB.p_);
const CVec2d nrmB = NrmB.Mat3Vec2_AffineDirection(matAffineB.p_);
const double lambda = rgd_v2m3::ResolveContact(aRS[irbA],aRS[irbB],
penetration, pA, pB, nrmB);
aContact.push_back({irbA,irbB,ip,ieB,reB,NrmB,lambda});
}
}
}
// finalize position, set velocity
for(CRigidState2& rs : aRS) {
if( rs.is_fix ) {
rs.velo = CVec2d(0,0);
rs.omega = 0.0;
continue;
}
rs.velo = (rs.posg_tmp - rs.posg)/dt;
rs.omega = (rs.theta_tmp - rs.theta)/dt;
rs.posg = rs.posg_tmp;
rs.theta = rs.theta_tmp;
}
// add frictional velocity change
for(CContactInfo2& c : aContact ){
CRigidState2& rbA = aRS[c.irbA];
CRigidState2& rbB = aRS[c.irbB];
assert( rbA.shape.size() == rbA.shape_velo.size() );
assert( rbB.shape.size() == rbB.shape_velo.size() );
const CVec2d& PA = aRS[c.irbA].shape[c.ipA];
const CVec2d& VA = aRS[c.irbA].shape_velo[c.ipA];
const unsigned int ieB = c.ieB;
const double reB = c.reB;
const unsigned int npB = static_cast<unsigned int>(rbB.shape.size());
const CVec2d PB = (1-reB)*rbB.shape[(ieB+0)%npB] + reB*rbB.shape[(ieB+1)%npB];
const CVec2d VB = (1-reB)*rbB.shape_velo[(ieB+0)%npB] + reB*rbB.shape_velo[(ieB+1)%npB];
const CVec2d vA = rbA.omega*(PA - rbA.posl).Rotate(rbA.theta + M_PI*0.5) + rbA.velo + VA.Rotate(rbA.theta);
const CVec2d vB = rbB.omega*(PB - rbB.posl).Rotate(rbB.theta + M_PI*0.5) + rbB.velo + VB.Rotate(rbB.theta);
const CVec2d pA = (PA - rbA.posl).Rotate(rbA.theta) + rbA.posg;
const CVec2d pB = (PB - rbB.posl).Rotate(rbB.theta) + rbB.posg;
const CVec2d tangent_dir = c.Njn.Rotate(rbB.theta - M_PI*0.5); // tangent direction
double velo_slip = (vA-vB).dot(tangent_dir); // slipping velocity
rgd_v2m3::AddFriction(rbA,rbB,
pA,pB,tangent_dir,velo_slip,c.lambda/dt);
}
 */
//}

fn main() -> anyhow::Result<()> {
    use del_fem_core::pbd_rigidbody2::RigidBody;
    let rb0 = RigidBody {
        vtx2xy: vec![
            -1.0, 0.0, 1.0, 0.0, 1.0, 0.8, 0.9, 0.8, 0.9, 0.2, -0.9, 0.2, -0.9, 0.8, -1.0, 0.8,
        ],
        pos_cg_ref: [0f32; 2],
        pos_cg_def: [0f32; 2],
        velo: [0f32; 2],
        theta: 0f32,
        is_fix: true,
        omega: 0f32,
        pos_cg_tmp: [0f32; 2],
        theta_tmp: 0f32,
        mass: 0f32,
        moment_of_inertia: 0f32,
    };
    let rb1 = RigidBody {
        vtx2xy: vec![0.0, 0.0, 0.4, 0.0, 0.4, 0.2, 0.0, 0.2],
        pos_cg_ref: [0f32; 2],
        pos_cg_def: [0f32, 0.5f32],
        velo: [0f32; 2],
        theta: std::f32::consts::PI * 0.1,
        is_fix: false,
        omega: 0f32,
        pos_cg_tmp: [0f32; 2],
        theta_tmp: 0f32,
        mass: 0f32,
        moment_of_inertia: 0f32,
    };
    let mut rbs = vec![rb0, rb1];
    for rb in rbs.iter_mut() {
        let rho = 1.0;
        let area = del_msh_cpu::polyloop2::area(&rb.vtx2xy);
        let cg = del_msh_cpu::polyloop2::cog_as_face(&rb.vtx2xy);
        let moment = del_msh_cpu::polyloop2::moment_of_inertia(&rb.vtx2xy, &cg);
        rb.mass = rho * area;
        rb.moment_of_inertia = rho * moment;
        rb.pos_cg_ref = cg;
        dbg!(cg);
    }
    //
    let image_shape = (300, 300);
    let mut canvas = del_canvas::canvas_gif::Canvas::new(
        std::path::Path::new("target/00_del_fem_canvas__rigid2.gif"),
        image_shape,
        &vec![0x112F41, 0xED553B, 0xF2B134, 0x068587],
    )?;
    let transform_world2pix: [f32; 9] = {
        let t0 = del_geo_core::aabb2::to_transformation_world2unit_ortho_preserve_asp(&[
            -1.2, -0.2, 1.2, 1.0,
        ]);
        let t1 = del_geo_core::mat3_col_major::from_transform_unit2pix(image_shape);
        del_geo_core::mat3_col_major::mult_mat_col_major(&t1, &t0)
    };

    for _itr in 0..200 {
        step_time(&mut rbs, 0.01, &[0.0, -10.0]);
        canvas.clear(0);
        for rb in rbs.iter() {
            let obj2world = rb.local2world();
            let obj2pix =
                del_geo_core::mat3_col_major::mult_mat_col_major(&transform_world2pix, &obj2world);
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
    Ok(())
}
