use del_geo_core::vec2;

#[derive(Clone)]
struct RigidBody {
    vtx2xy: Vec<f32>,
    pos: [f32; 2],
    velo: [f32; 2],
    theta: f32,
    omega: f32,
    is_fix: bool,
    pos_tmp: [f32; 2],
    theta_tmp: f32,
    mass: f32,
    I: f32
}

impl RigidBody {
    fn local2world(&self) -> [f32; 9] {
        let r0 = del_geo_core::mat3_col_major::from_rotate(self.theta);
        let t0 = del_geo_core::mat3_col_major::from_translate(&self.pos);
        del_geo_core::mat3_col_major::mult_mat_col_major(&t0, &r0)
    }
}


fn polygon2_sdf(
    shape: &Vec<f32>,
    q: &[f32; 2]) -> (f32, [f32; 2])
{
    use del_geo_core::vec2;
    let nej = shape.len() / 2;
    let mut min_dist = -1.0;
    let mut winding_number = 0f32;
    let mut pos_near = [0f32; 2];
    let mut ie_near = 0;
    for iej in 0..nej {
        let ps = arrayref::array_ref!(shape, ((iej + 0) % nej)*2, 2);
        let pe = arrayref::array_ref!(shape, ((iej + 1) % nej)*2, 2);
        winding_number += del_geo_core::edge2::winding_number(&ps, &pe, q);
        let (_rm, pm) = del_geo_core::edge2::nearest_point2(ps, pe, q);
        let dist0 = del_geo_core::edge2::length(&pm, q);
        if dist0 == 0.0 {
            dbg!("hoge", pm, ps, pe, q);
        }
        if min_dist > 0. && dist0 > min_dist { continue; }
        min_dist = dist0;
        pos_near = pm;
        ie_near = iej;
    }
    //
    let normal_out = {
        // if distance is small use edge's normal
        let ps = arrayref::array_ref!(shape, ((ie_near + 0) % nej)*2, 2);
        let pe = arrayref::array_ref!(shape, ((ie_near + 1) % nej)*2, 2);
        let ne = vec2::sub(pe, ps);
        let ne = vec2::rotate(&ne, -std::f32::consts::PI * 0.5);
        vec2::normalize(&ne)
    };
    //
    // dbg!(winding_number);
    if (winding_number - 1.0).abs() < 0.5 { // inside
        let normal = if min_dist < 1.0e-5 {
            normal_out
        } else {
            vec2::normalize(&vec2::sub(&pos_near, q))
        };
        (-min_dist, normal)
    } else {
        let normal = if min_dist < 1.0e-5 {
            normal_out
        } else {
            vec2::normalize(&vec2::sub(q, &pos_near))
        };
        (min_dist, normal)
    }
}

#[test]
fn test_polygon2_sdf() {
    let vtx2xy = vec!(
        0., 0.,
        1.0, 0.0,
        1.0, 0.2,
        0.0, 0.2);
    use del_geo_core::vec2;
    {
        let (sdf, normal) = polygon2_sdf(&vtx2xy, &[0.01, 0.1]);
        assert!((sdf + 0.01).abs() < 1.0e-5);
        assert!(vec2::length(&vec2::sub(&normal, &[-1., 0.])) < 1.0e-5);
    }
    {
        let (sdf, normal) = polygon2_sdf(&vtx2xy, &[-0.01, 0.1]);
        assert!((sdf - 0.01).abs() < 1.0e-5);
        assert!(vec2::length(&vec2::sub(&normal, &[-1., 0.])) < 1.0e-5);
    }
}

fn ResolveContact(
    rbA: &mut RigidBody,
    rbB: &mut RigidBody,
    penetration: f32,
    pA: &[f32; 2],
    pB: &[f32; 2],
    nB: &[f32; 2]) -> f32
{
    use del_geo_core::vec2;
    let mut deno = 0f32;
    if !rbA.is_fix {
        deno += 1. / rbA.mass;
        let t0 = vec2::area_quadrilateral(&vec2::sub(pA, &rbA.pos), nB);
        deno += t0 * t0 / rbA.I;
    }
    if !rbB.is_fix {
        deno += 1. / rbB.mass;
        let t0 = vec2::area_quadrilateral(&vec2::sub(pB, &rbB.pos), &[-nB[0], -nB[1]]);
        deno += t0 * t0 / rbB.I;
    }
    let lambda = penetration / deno; // force*dt*dt
    if !rbA.is_fix {
        rbA.pos_tmp = [
            rbA.pos_tmp[0] + (lambda / rbA.mass) * nB[0],
            rbA.pos_tmp[1] + (lambda / rbA.mass) * nB[1] ];
        let tA = vec2::area_quadrilateral(&vec2::sub(pA, &rbA.pos), &nB);
        rbA.theta_tmp += tA * lambda / rbA.I;
        // dbg!(pA, rbA.pos, tA, lambda, rbA.I);
    }
    if !rbB.is_fix {
        rbB.pos_tmp = [
            rbB.pos_tmp[0] - (lambda / rbB.mass) * nB[0],
            rbB.pos_tmp[1] - (lambda / rbB.mass) * nB[1] ];
        let tB = vec2::area_quadrilateral(&vec2::sub(pB, &rbB.pos), &[-nB[0], -nB[1]]);
        // dbg!(tB, lambda, rbB.I);
        rbB.theta_tmp += tB * lambda / rbB.I;
    }
    lambda
}

fn step_time(
    rbs: &mut [RigidBody],
    dt: f32,
    gravity: &[f32; 2])
{
    for rb in rbs.iter_mut() {
        if rb.is_fix {
            rb.pos_tmp = rb.pos;
            rb.theta_tmp = rb.theta;
            continue;
        }
        rb.velo = [
            rb.velo[0] + gravity[0] * dt,
            rb.velo[1] + gravity[1] * dt];
        rb.pos_tmp = [
            rb.pos[0] + dt * rb.velo[0],
            rb.pos[1] + dt * rb.velo[1]];
        rb.theta_tmp = rb.theta + dt * rb.omega;
    }
    //

    use del_geo_core::mat3_col_major;
    for irb in 0..rbs.len() {
        let lcli2world = rbs[irb].local2world();
        dbg!(rbs[irb].theta);
        for jrb in 0..rbs.len() {
            if irb == jrb { continue; }
            let lclj2world = rbs[jrb].local2world();
            let lcli2lclj = mat3_col_major::mult_mat_col_major(
                &mat3_col_major::try_inverse(&lclj2world).unwrap(),
                &lcli2world);
            for ivtx in 0..rbs[irb].vtx2xy.len() / 2 {
                let xyi = arrayref::array_ref!(rbs[irb].vtx2xy, ivtx*2, 2);
                let xyi_lclj = mat3_col_major::transform_homogeneous(&lcli2lclj, xyi).unwrap();
                let (sdf, nrm) = polygon2_sdf(&rbs[jrb].vtx2xy, &xyi_lclj);
                if sdf > 0.0 { continue; }
                dbg!(sdf);
                let xyi_world = mat3_col_major::transform_homogeneous(&lcli2world, &xyi).unwrap();
                let nrmB = mat3_col_major::transform_direction(&lcli2world, &nrm);
                let mut rbi = rbs[irb].clone();
                let mut rbj = rbs[jrb].clone();
                let lambda = ResolveContact(
                    &mut rbi, &mut rbj,
                    -sdf, &xyi_world, &xyi_world, &nrmB);
                rbs[irb] = rbi;
                rbs[jrb] = rbj;
            }
        }
    }

    // finalize position, set velocity
    for rb in rbs.iter_mut() {
        if rb.is_fix {
            rb.velo = [0f32; 2];
            rb.omega = 0.0;
            continue;
        }
        rb.velo = [
            (rb.pos_tmp[0] - rb.pos[0]) / dt,
            (rb.pos_tmp[1] - rb.pos[1]) / dt];
        rb.omega = (rb.theta_tmp - rb.theta) / dt;
        rb.pos = rb.pos_tmp;
        rb.theta = rb.theta_tmp;
    }
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

fn main() {
    let mut rb0 = RigidBody {
        vtx2xy: vec!(
            -1.0, 0.0,
            1.0, 0.0,
            1.0, 0.8,
            0.9, 0.8,
            0.9, 0.2,
            -0.9, 0.2,
            -0.9, 0.8,
            -1.0, 0.8),
        pos: [0f32; 2],
        velo: [0f32; 2],
        theta: 0f32,
        is_fix: true,
        omega: 0f32,
        pos_tmp: [0f32; 2],
        theta_tmp: 0f32,
        mass: 0f32,
        I: 0f32
    };
    let mut rb1 = RigidBody {
        vtx2xy: vec!(
            0.0, 0.0,
            0.4, 0.0,
            0.4, 0.2,
            0.0, 0.2),
        pos: [0f32, 0.5f32],
        velo: [0f32; 2],
        theta: std::f32::consts::PI * 0.3,
        is_fix: false,
        omega: 0f32,
        pos_tmp: [0f32; 2],
        theta_tmp: 0f32,
        mass: 0f32,
        I: 0f32
    };
    let mut rbs = vec!(rb0, rb1);
    for rb in rbs.iter_mut() {
        let rho = 1.0;
        let area = del_msh_core::polyloop2::area_(&rb.vtx2xy);
        let cg = del_msh_core::polyloop2::cog_as_face(&rb.vtx2xy);
        let moment = del_msh_core::polyloop2::RotationalMomentPolar_Polygon2(&rb.vtx2xy, &cg);
        rb.mass = rho * area;
        rb.I = rho * moment;
    }
    //
    let image_shape = (300, 300);
    let mut canvas = del_canvas_cpu::canvas_gif::Canvas::new(
        std::path::Path::new("target/00_del_fem_canvas__rigid2.gif"),
        image_shape,
        &vec![0x112F41, 0xED553B, 0xF2B134, 0x068587],
    );
    let transform_world2pix: [f32; 9] = {
        let t0 = del_geo_core::aabb2::to_transformation_world2unit_ortho_preserve_asp(&[-1.2, -0.2, 1.2, 1.0]);
        let t1 = del_geo_core::mat3_col_major::transform_unit2pix(image_shape);
        del_geo_core::mat3_col_major::mult_mat_col_major(&t1, &t0)
    };

    for itr in 0..30 {
        step_time(&mut rbs, 0.01, &[0.0, -10.0]);
        canvas.clear(0);
        for rb in rbs.iter() {
            let obj2world = rb.local2world();
            let obj2pix = del_geo_core::mat3_col_major::mult_mat_col_major(
                &transform_world2pix,
                &obj2world);
            del_canvas_cpu::rasterize_polygon::stroke::<f32, u8>(
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