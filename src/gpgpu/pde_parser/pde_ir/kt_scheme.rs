use crate::gpgpu::pde_parser::pde_ir::{ir_helper::*, SPDETokens::*, *};

pub fn kt(
    u: &SPDETokens,
    fu: &SPDETokens,
    eigs: &Vec<SPDETokens>,
    theta: f64,
    d: usize,
) -> SPDETokens {
    let iv = Symb(["ivdx", "ivdy", "ivdz"][d].into());
    (h(u, fu, eigs, theta, d, 1) - h(u, fu, eigs, theta, d, -1)) * iv
}

pub fn h(
    u: &SPDETokens,
    fu: &SPDETokens,
    eigs: &Vec<SPDETokens>,
    theta: f64,
    d: usize,
    idir: i32,
) -> SPDETokens {
    let p = &idx(1, idir, d);
    let m = &idx(-1, idir, d);
    let min = |a, b| func("min", vec![a, b]);
    let max = |a, b| func("max", vec![a, b]);
    let abs = |a| func("fabs", vec![a]);
    let sign = |a| func("sign", vec![a]);
    let minmod = |a: SPDETokens, b: SPDETokens| {
        (sign(a.clone()) + sign(b.clone())) * Const(0.5) * min(abs(a), abs(b))
    };
    let ud = |i: i32| {
        let mut id = [0; 4];
        id[d] = i;
        move |u: Indexable| Indx(u.apply_idx(&id))
    };
    let ux = |u: Indexable| {
        let u = Indx(u);
        let up = ap(&u, ud(1));
        let um = ap(&u, ud(-1));
        let uc = u;
        minmod(
            Const(theta) * (uc.clone() - um.clone()),
            minmod((up.clone() - um) * Const(0.5), Const(theta) * (up - uc)),
        )
    };
    let up = |u: Indexable| {
        let u = u.clone().apply_idx(p);
        Indx(u.clone()) - ux(u) * Const(0.5)
    };
    let um = |u: Indexable| {
        let u = u.clone().apply_idx(m);
        Indx(u.clone()) + ux(u) * Const(0.5)
    };
    let a = |eigs: &Vec<SPDETokens>| {
        if eigs.len() > 0 {
            let mut tmp = eigs
                .iter()
                .map(|e| max(abs(ap(e, up)), abs(ap(e, um))))
                .collect::<Vec<_>>();
            let fst = tmp
                .pop()
                .expect("There must be at least one eigenvalue given for KT scheme.");
            tmp.into_iter().fold(fst, |acc, i| max(acc, i))
        } else {
            panic!("No eigenvalues provided.")
        }
    };
    (ap(fu, up) + ap(fu, um) - a(eigs) * (ap(u, up) - ap(u, um))) * Const(0.5)
}
fn ap(u: &SPDETokens, f: impl Fn(Indexable) -> SPDETokens) -> SPDETokens {
    u.clone().apply_indexable(&f)
}

fn idx(dir: i32, idir: i32, d: usize) -> [i32; 4] {
    let i = (dir + idir) / 2;
    [[i, 0, 0, 0], [0, i, 0, 0], [0, 0, i, 0]][d]
}
