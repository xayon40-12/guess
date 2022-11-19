use crate::concurrent_hdf5::ConcurrentHDF5;
use crate::gpgpu::algorithms::ReduceParam;
use crate::gpgpu::algorithms::{
    moments_to_cumulants,
    AlgorithmParam::{self, *},
    MomentsParam,
};
use crate::gpgpu::descriptors::KernelArg::*;
use crate::gpgpu::kernels::{radial, Origin, Radial};
use crate::gpgpu::{Dim::*, DimDir::*};
use crate::simulation::Vars;
use ndarray::Array;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;

#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub enum Shape {
    All,
    Radial,
}
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum WindowOption {
    All,
    Exact(Vec<f64>),
    Every(f64),
}
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum Action {
    Moments(Vec<String>),
    // StaticStructureFactor(Vec<String>, Shape),
    // DynamicStructureFactor(Vec<String>),
    Correlation(Vec<String>, Shape),
    RawData(Vec<String>),
    Window(Vec<String>, WindowOption), // buffers
    Anisotropy(Vec<(String, String)>), // the strings correspond to the two values that should averaged over space and compared
}
pub use Action::*;

pub type Callback = Box<
    dyn FnMut(
        &mut crate::gpgpu::Handler,
        &Vars,
        &mut Option<ConcurrentHDF5>,
        f64,
    ) -> crate::gpgpu::Result<()>,
>;

fn write_all<'a>(parent: &'a str, file_name: &'a str, content: &'a str) {
    let write = |f, c: &str| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&format!("{}/{}", parent, f))?
            .write_all(c.as_bytes())
    };
    if let Err(e) = write(file_name, content) {
        eprintln!("Could not write to file \"{}\".\n{:?}", file_name, e);
    }
}

macro_rules! gen {
    ($names:ident, $id:ident, $head:ident, $name_to_index:ident, $num_pdes:ident, $h:ident, $vars:ident, $hdf5_file:ident, $t:ident, $body:tt) => {{
        let idx = $names.iter().map(|n| $name_to_index[n]).collect::<Vec<_>>();
        Box::new(move |$h, $vars, $hdf5_file, $t| {
            for $id in idx.iter().map(|&i| i) {
                $body
            }

            Ok(())
        })
    }};
}
macro_rules! gen_pair {
    ($names:ident, $id:pat, $head:ident, $name_to_index:ident, $num_pdes:ident, $h:ident, $vars:ident, $hdf5_file:ident, $t:ident, $body:tt) => {{
        let idx_pair = $names
            .iter()
            .map(|(n1, n2)| ($name_to_index[n1], $name_to_index[n2]))
            .collect::<Vec<_>>();
        Box::new(move |$h, $vars, $hdf5_file, $t| {
            for $id in idx_pair.iter().map(|&i| i) {
                $body
            }

            Ok(())
        })
    }};
}

fn strip(s: &str) -> String {
    s.replace("dvar_", "")
        .replace("swap_", "")
        .replace("constraint_", "")
}

// This function might be used with same first and second buffer as it compute only order 1 and 2
fn moms(
    w: u32,
    vars: &Vars,
    bufs: &[&'static str],
    h: &mut crate::gpgpu::Handler,
    complex: bool,
) -> crate::gpgpu::Result<Vec<Vec<f64>>> {
    let mut dim: [usize; 3] = vars.dim.into();
    let mut dirs = vars.dim.all_dirs();
    dirs.retain(|v| !vars.dirs.contains(v));
    let mul = if complex { 2 } else { 1 };
    let prm = MomentsParam {
        num: 2,
        vect_dim: w * mul,
        packed: false,
    };
    h.run_algorithm("moments", dim.into(), &dirs, bufs, Ref(&prm))?;
    dirs.iter().for_each(|d| dim[*d as usize] = 1);
    let len = dim[0] * dim[1] * dim[2];
    h.run_arg(
        "to_var",
        D1(len * w as usize * mul as usize),
        &[BufArg(bufs[3], "src")],
    )?;
    let res = h.get_firsts(bufs[3], len * 2 * w as usize)?;
    let res = if complex {
        unsafe {
            let mut res: Vec<f64> = std::mem::transmute(res.VF64_2());
            res.set_len(2 * len * 2 * w as usize);
            res
        }
    } else {
        res.VF64()
    };
    Ok(res
        .chunks(len * w as usize * mul as usize)
        .map(|i| i.iter().map(|v| *v).collect::<Vec<f64>>())
        .collect::<Vec<_>>())
}
fn vtos<T, F: Fn(&T) -> String>(v: &[T], f: F) -> String {
    v.iter().map(|i| f(i)).collect::<Vec<_>>().join(" ")
}
fn vvtos<T, F: Fn(&[T]) -> String>(v: &[T], w: usize, f: F) -> String {
    v.chunks(w).map(|i| f(i)).collect::<Vec<_>>().join(" ")
}
fn venot(v: &[f64]) -> String {
    v.iter().map(enot).collect::<Vec<_>>().join(";")
}
fn enot(v: &f64) -> String {
    format!("{:e}", v)
}
fn renot(v: &Radial) -> String {
    format!("{:e}:{}", v.pos, venot(&v.vals)) // <coord>;<number>
}
// fn cenot(v: &[f64; 2]) -> String {
//     format!("{:e}j{:e}", v[0], v[1]) // complex number <real>j<imaginary>
// }

macro_rules! attr {
    ($storage:ident, $parent:expr, $t:expr, $observable:expr, $attr:expr, $v:expr) => {
        let location = &format!("{}/t/{:e}/{}", $parent, $t, $observable);
        if let Err(e) = $storage.update_group_attr(location, $attr, |i| i, $v) {
            eprintln!(
                "Error in setting \"{}\" attribute in hdf5 file:\n{:?}",
                $attr, e
            );
        }
    };
}
macro_rules! hdf5write {
    ($storage:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, m $data:expr) => {
        hdf5write!($storage, $parent, $t, $observable, $name, m $data, shapex 1);
    };
    ($storage:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, $data:expr) => {
        hdf5write!($storage, $parent, $t, $observable, $name, $data, shapex 1);
    };
    ($storage:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, m $data:expr, shapex $x:expr) => {
        for (mom_id, data) in $data.iter().enumerate() {
            hdf5write!($storage, $parent, $t, &format!("{}/moment/{}", $observable, mom_id+1), $name, data, shapex $x);
        }
    };
    ($storage:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, $data:expr, shapex $x:expr) => {
        let location = &format!("{}/t/{:e}/{}", $parent, $t, $observable);
        let x = ($x) as usize;
        let err = if x == 1 {
            $storage.write_data(location, $name, $data)
        } else {
            let y = $data.len()/x;
            let data_vals = Array::from_shape_vec((y, x), $data.clone())
                .expect("Wrong array shape in Window hdf5 storing");
            $storage.write_data(location, $name, &data_vals)
        };
        if let Err(e) = err {
            eprintln!(
                "Error in writing data at \"{}\" in hdf5 file:\n{:?}",
                location, e
            );
        }
    };
}
macro_rules! save_radial {
    ($hdf5:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, $data:ident, shapex $w:expr) => {
        let data_pos = $data.iter().map(|r| r.pos).collect::<Vec<_>>();
        let data_vals = $data.iter().flat_map(|r| r.vals.clone()).collect::<Vec<_>>();
        hdf5write!($hdf5, $parent, $t, $observable, $name, &data_vals, shapex $w);
        hdf5write!($hdf5, $parent, $t, $observable, "coord", &data_pos);
    };
    ($hdf5:ident, $parent:expr, $t:expr, $observable:expr, $name:expr, m $data:ident, shapex $w:expr) => {
        let data_pos = $data[0].iter().map(|r| r.pos).collect::<Vec<_>>();
        let data_vals = $data.iter().map(|r| r.iter().flat_map(|r| r.vals.clone()).collect::<Vec<_>>()).collect::<Vec<_>>();
        hdf5write!($hdf5, $parent, $t, $observable, $name, m &data_vals, shapex $w);
        hdf5write!($hdf5, $parent, $t, $observable, "coord", &data_pos);
    };
}

impl Action {
    pub fn to_callback(
        &self,
        name_to_index: &HashMap<String, usize>,
        _num_pdes: usize,
    ) -> Callback {
        match self {
            Window(names, positions) => {
                let positions = positions.clone();
                gen! {names,id,head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
                    let var_name = strip(&vars.dvars[id].0);
                    let w = vars.dvars[id].1;
                    let dim: [usize; 3] = vars.dim.into();
                    let phy: [f64; 3] = vars.phy;
                    let len = dim[0]*dim[1]*dim[2];
                    let dx3 = dim.iter().zip(phy.iter()).filter(|&(_,&f)| f != 0.0).fold(1.0, |acc,(&i,&f)| acc*f/i as f64);
                    let raw = h.get_firsts(&vars.dvars[id].0, len*w as usize)?.VF64();
                    let mut rad = radial(&raw, w as usize, &dim, &phy, false, Origin::Center, true);
                    let vl = rad[0][0].vals.len();
                    rad.par_iter_mut().for_each(|r|
                        for i in 1..r.len() {
                            for j in 0..vl {
                                r[i].vals[j] += r[i-1].vals[j];
                            }
                        }
                    );
                    rad.par_iter_mut().for_each(|r| // mutiply by dx*dy*dz to actually integrate and not only sum
                        for i in r.iter_mut().skip(1) {
                            for j in 0..vl {
                                i.vals[j] *= dx3;
                            }
                        }
                    );
                    let num = 4; //number of moments
                    let configurations = rad.len();
                    let search = |v: &Vec<Radial>| {
                        match &positions {
                            WindowOption::Exact(poss) => {
                                poss.iter().filter_map(|p| v.iter().find(|x| x.pos >= *p).and_then(|x| Some(x.clone()))).collect()
                            },
                            WindowOption::Every(e) => {
                                (0..(v[v.len()-1].pos/e + 1.0) as usize).filter_map(|p| v.iter().find(|x| x.pos >= p as f64 * e).and_then(|x| Some(x.clone()))).collect()
                            },
                            WindowOption::All => {
                                v.clone()
                            },
                        }
                    };

                    let observable = "radial_window";
                    if let Some(hdf5) = hdf5_file {
                        if configurations > 1 {
                            let data = moments(rad,num).into_iter().map(|m| search(&m)).collect::<Vec<_>>();
                            save_radial!(hdf5, vars.parent, t, observable, &var_name, m data, shapex w);
                            attr!(hdf5, vars.parent, t, observable, "noise_configurations", &configurations);
                        } else {
                            let data = search(&rad[0]);
                            save_radial!(hdf5, vars.parent, t, observable, &var_name, data, shapex w);
                        }
                    } else {
                        let moms = if configurations > 1 {
                            moments(rad,num).into_iter().map(|m| vtos(&search(&m),renot)).collect::<Vec<_>>().join("/")
                        } else {
                            vtos(&search(&rad[0]),renot)
                        };
                        write_all(&vars.parent, "radial_window.txt", &format!("{:e}|{}|{}#{}\n", t, var_name, configurations, moms));
                    }
                }}
            }
            Moments(names) => gen! {names,id,head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
                let var_name = strip(&vars.dvars[id].0);
                let w = vars.dvars[id].1;
                let num = 4;
                let prm = MomentsParam{ num: num as _, vect_dim: w, packed: true };
                h.run_algorithm("moments", vars.dim, &vars.dirs, &[&vars.dvars[id].0,"tmp","sum","tmp2"], Ref(&prm))?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    vars.dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    let len = dim[0]*dim[1]*dim[2];
                    let prm = MomentsParam{ num: num as _, vect_dim: w, packed: false };
                    //h.run_arg("moments_to_cumulants", D1(len), &[Buffer("moments"),Buffer("cumulants"),Param("vect_dim",w.into()),Param("num",(num as u32).into())])?;
                    h.run_algorithm("moments", D2(num,len), &[Y], &["tmp2","tmp","sum","tmp3"], Ref(&prm))?;// WARNING use "cumulants" as tmp buffer
                    //h.run_arg("to_var",D1(num*w as usize),&[BufArg("tmp3","src")])?;
                    let res = h.get_firsts("tmp3",num*num*w as usize)?.VF64();
                    //let cumulants = vtos(&moments_to_cumulants(&res[0..(num*w as usize)],w as _),enot);
                    //let moms = res.chunks(num*w as usize) // use moments
                    //    .map(|c| vvtos(c,w as usize,venot))
                    //    .collect::<Vec<String>>();

                    if let Some(hdf5) = hdf5_file {
                        let res = res.chunks(num*w as usize).map(|c| c.iter().map(|v| *v).collect::<Vec<_>>()).collect::<Vec<_>>();
                        hdf5write!(hdf5, vars.parent, t, "moments", &var_name, m &res, shapex w);
                        attr!(hdf5, vars.parent, t, "moments", "noise_configurations", &len);
                    } else {
                        let moms = vvtos(&res, w as usize, venot);
                        write_all(&vars.parent, "moments.txt", &format!("{:e}|{}|{}#{}\n", t, var_name, len, moms));
                    }
                    //write_all(&vars.parent, "moments.txt", &format!("{:e}|{}|sigma_moments|{}\n", t, var_name,&moms[1]));
                    //write_all(&vars.parent, "moments.txt", &format!("{:e}|{}|cumulants|{}\n", t, var_name,&cumulants));

                } else {
                    let moments = h.get_firsts("moments",num*w as usize)?.VF64();
                    //let cumulants = moments_to_cumulants(&moments, w as _);
                    if let Some(hdf5) = hdf5_file {
                        hdf5write!(hdf5, vars.parent, t, "moments", &var_name, &moments, shapex w);
                    } else {
                        write_all(&vars.parent, "moments.txt", &format!("{:e}|{}|{}#{}\n", t, var_name, 1, vvtos(&moments,w as usize,venot)));
                    }
                    //write_all(&vars.parent, "moments.txt", &format!("{:e}|{}|cumulants|{}\n", t, var_name,vvtos(&cumulants,w as usize,venot)));
                }
            }},
            // StaticStructureFactor(names, shape) => {
            //     let shape = *shape;
            //     gen! {names,id,head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
            //     let var_name = strip(&vars.dvars[id].0);
            //         let w = vars.dvars[id].1;
            //         let len = vars.len;
            //         let num = 4;
            //         h.run_arg("complex_from_real", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("srcFFT","dst")])?;
            //         h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], Ref(&w))?;
            //         h.run_arg("kc_sqrmod", D1(len*w as usize), &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;
            //         let phy = vars.phy.iter().fold(1.0, |a,i| if *i == 0.0 { a } else { i*a });
            //         h.run_arg("ctimes", D1(len*w as usize), &[BufArg("tmp","src"),Param("c",phy.into()),BufArg("tmp","dst")])?;
            //         let dim: [usize;3] = vars.dim.into();
            //         let phy = vars.phy;
            //         match shape {
            //             Shape::All => {
            //                 if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
            //                     let moms = moms(w,&vars,&["tmp","tmp","sum","tmp3"],h,false)?;
            //                     let (moms,name) = (moms.iter().map(|i| vvtos(i,w as usize,venot)).collect::<Vec<_>>(), "SF");
            //                     write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|{}|{}\n", t, var_name, name, &moms[0]));
            //                     //write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|sigma_{}|{}\n", t, var_name, name, &moms[1]));
            //                 } else {
            //                     let moms = h.get_firsts("tmp",len*w as usize)?.VF64();
            //                     let (moms,name) = (vvtos(&moms,w as usize,venot),"SF");
            //                     write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|{}|{}\n", t, var_name, name,moms));
            //                 }
            //             },
            //             Shape::Radial => {
            //                 let a = h.get_firsts("tmp", len*w as usize)?.VF64();
            //                 let rad = radial(&a, w as usize, &dim, &phy, true, Origin::Corner, false);
            //                 let configurations = rad.len();
            //                 if configurations > 1 {
            //                     let rad = moments(rad,num); // use moments and not cumulants
            //                     let (moms,name) = (rad.into_iter().map(|m| vtos(&m,renot)).collect::<Vec<_>>().join("/"),"radial_SF");
            //                     write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|{}|{}#{}\n", t, var_name, name, configurations, moms));
            //                     //write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|sigma_{}|{}\n", t, var_name, name, &moms[1]));
            //                 } else {
            //                     let (moms,name) = (vtos(&rad[0],renot), "radial_SF");
            //                     write_all(&vars.parent, "static_structure_factor.txt", &format!("{:e}|{}|{}|{}\n", t, var_name, name,moms));
            //                 }
            //             },

            //         }
            //     }}
            // }
            // DynamicStructureFactor(names) => {
            //     let mut first = true;
            //     gen! {names,id,head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
            //     let var_name = strip(&vars.dvars[id].0);
            //         let w = vars.dvars[id].1;
            //         let len = vars.len;
            //         h.run_arg("complex_from_real", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("srcFFT","dst")])?;
            //         h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], Ref(&w))?;
            //         if first {
            //             first = false;
            //             h.copy("dstFFT","initFFT")?;
            //         }
            //         h.run_arg("kc_times_conj", D1(len*w as usize), &[BufArg("initFFT","a"),BufArg("dstFFT","b"),BufArg("dstFFT","dst")])?;
            //         let phy = vars.phy.iter().fold(1.0, |a,i| if *i == 0.0 { a } else { i*a });
            //         h.run_arg("ctimes", D1(len*w as usize*2), &[BufArg("dstFFT","src"),Param("c",phy.into()),BufArg("dstFFT","dst")])?;
            //         if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
            //             let moms = moms(w,vars,&["dstFFT","dstFFT","srcFFT","tmpFFT"],h,true)?;
            //             write_all(&vars.parent, "dynamic_structure_factor.txt", &format!("{:e}|{}|DSF|{}\n", t, var_name, vvtos(&moms[0],w as usize,venot)));
            //             //write_all(&vars.parent, "dynamic_structure_factor.txt", &format!("{:e}|{}|sigma_DSF|{}\n", t, var_name, vvtos(&moms[1],w as usize,venot)));
            //         } else {
            //             let moms = h.get_firsts("dstFFT",len*w as usize)?.VF64_2();
            //             write_all(&vars.parent, "dynamic_structure_factor.txt", &format!("{:e}|{}|DSF|{}\n", t, var_name, vtos(&moms,cenot)));
            //         }
            //     }}
            // }
            Correlation(names, shape) => {
                let shape = *shape;
                gen! {names,id,head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
                    let var_name = strip(&vars.dvars[id].0);
                    let w = vars.dvars[id].1;
                    let len = vars.len;
                    let num = 4;
                    //let prm = MomentsParam{ num: 1, vect_dim: w, packed: true };
                    let dim: [usize; 3] = vars.dim.into();
                    let mut dim: Vec<u32> = dim.iter().map(|&x| x as u32).collect();
                    vars.dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    //h.run_algorithm("moments", vars.dim, &vars.dirs, &[&vars.dvars[id].0,"tmp","sum","moments"], Ref(&prm))?;
                    //h.run_arg("vcminus", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("moments","c"),BufArg("sum","dst"),Param("size",[dim[0],dim[1],dim[2],w].into()),Param("vect_dim",w.into())])?;
                    //h.run_algorithm("correlation",vars.dim,&vars.dirs,&["sum","tmp"],Ref(&w))?;
                    h.run_algorithm("correlation",vars.dim,&vars.dirs,&[&vars.dvars[id].0,"tmp"],Ref(&w))?;
                    let dim: [usize;3] = vars.dim.into();
                    let phy = vars.phy;

                    match shape {
                        Shape::All => {
                            if let Some(hdf5) = hdf5_file {
                                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                                    let moms = moms(w,vars,&["tmp","tmp","sum","tmp3"],h,false)?; // FIXME: the moms function automatically compute the variance which is not consistent with the other variables where simply the moments a computed
                                    hdf5write!(hdf5, vars.parent, t, "correlation", &var_name, m moms, shapex w);
                                    attr!(hdf5, vars.parent, t, "correlation", "noise_configurations", &len);
                                } else {
                                    let moms = h.get_firsts("tmp",len*w as usize)?.VF64();
                                    hdf5write!(hdf5, vars.parent, t, "correlation", &var_name, &moms, shapex w);
                                }
                            } else {
                                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                                    let moms = moms(w,vars,&["tmp","tmp","sum","tmp3"],h,false)?;
                                    let moms = moms.iter().map(|i| vvtos(i,w as usize,venot)).collect::<Vec<_>>();
                                    write_all(&vars.parent, "correlation.txt", &format!("{:e}|{}|{}\n", t, var_name, &moms[0]));
                                    //write_all(&vars.parent, "correlation.txt", &format!("{:e}|{}|sigma_{}|{}\n", t, var_name, name, &moms[1]));
                                } else {
                                    let moms = h.get_firsts("tmp",len*w as usize)?.VF64();
                                    let moms = vvtos(&moms,w as usize,venot);
                                    write_all(&vars.parent, "correlation.txt", &format!("{:e}|{}|{}\n", t, var_name,moms));
                                }
                            }
                        },
                        Shape::Radial => {
                            let a = h.get_firsts("tmp", len*w as usize)?.VF64();
                            let rad = radial(&a, w as usize, &dim, &phy, true, Origin::Center, true);
                            let configurations = rad.len();
                            if let Some(hdf5) = hdf5_file {
                                if configurations > 1 {
                                    let data = moments(rad,num); //  .into_iter().map(|m| search(&m)).collect::<Vec<_>>();
                                    save_radial!(hdf5, vars.parent, t, "radial_correlation", &var_name, m data, shapex w);
                                    attr!(hdf5, vars.parent, t, "radial_correlation", "noise_configurations", &configurations);
                                } else {
                                    let data = &rad[0];
                                    save_radial!(hdf5, vars.parent, t, "radial_correlation", &var_name, data, shapex w);
                                }
                            } else {
                                if configurations > 1 {
                                    let rad = moments(rad,num); // use moments and not cumulants
                                    let moms = rad.into_iter().map(|m| vtos(&m,renot)).collect::<Vec<_>>().join("/");
                                    write_all(&vars.parent, "correlation.txt", &format!("{:e}|radial|{}|{}#{}\n", t, var_name, configurations, moms));
                                    //write_all(&vars.parent, "correlation.txt", &format!("{:e}|{}|sigma_{}|{}\n", t, var_name, name, &moms[1]));
                                } else {
                                    let moms = vtos(&rad[0],renot);
                                    write_all(&vars.parent, "correlation.txt", &format!("{:e}|radial|{}|{}\n", t, var_name ,moms));
                                }
                            }
                        },

                    }

                }}
            }
            RawData(names) => gen! {names,id,_head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
                let w = vars.dvars[id].1;
                let var_name = strip(&vars.dvars[id].0);
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let num = 4;
                    let dir = [X,Y,Z].iter().take(vars.dim.len()).filter(|i| !vars.dirs.contains(i)).map(|i| *i).collect::<Vec<crate::gpgpu::DimDir>>();
                    let mut dim: [usize;3] = vars.dim.into();
                    dir.iter().for_each(|d| dim[*d as usize] = 1);
                    let len = dim[0]*dim[1]*dim[2];
                    let prm = MomentsParam{ num, vect_dim: w, packed: false };
                    h.run_algorithm("moments", vars.dim, &dir, &[&vars.dvars[id].0,"tmp","tmp2","sum"], Ref(&prm))?;
                    let res = h.get_firsts("sum",num as usize*len*w as usize)?.VF64();
                    //let cumulants = moments_to_cumulants(&res,len*w as usize);//.chunks((num*w) as _).fold(vec![vec![];num as _], |mut acc,i| {i.chunks(w as _).enumerate().for_each(|(i,v)| acc[i].extend_from_slice(v)); acc});
                    //let moms = cumulants.chunks(len*w as usize)
                    if let Some(hdf5) = hdf5_file {
                        let res = res.chunks(len*w as usize).map(|c| c.iter().map(|v| *v).collect()).collect::<Vec<Vec<_>>>();
                        hdf5write!(hdf5, vars.parent, t, "raw", &var_name, m &res, shapex w);
                        attr!(hdf5, vars.parent, t, "raw", "noise_configurations", &len);
                    } else {
                        let moms = res.chunks(len*w as usize) // keep moments
                            .map(|c| vvtos(c,w as usize,venot))
                            .collect::<Vec<String>>();
                        let configurations = moms.len();
                        let moms = moms.join("/");
                        write_all(&vars.parent, "raw.txt", &format!("{:e}|{}|{}#{}\n", t, var_name, configurations, moms));
                    }
                } else {
                    let len = vars.len;
                    let raw = h.get_firsts(&vars.dvars[id].0,len*w as usize)?.VF64();
                    if let Some(hdf5) = hdf5_file {
                        hdf5write!(hdf5, vars.parent, t, "raw", &var_name, &raw, shapex w);
                    } else {
                        write_all(&vars.parent, "raw.txt", &format!("{:e}|{}|{}\n", t, var_name,
                                vvtos(&raw,w as usize,venot)
                        ));
                    }
                }
            }},
            Anisotropy(pairs) => {
                gen_pair! {pairs,(id1,id2),head,name_to_index,num_pdes,h,vars,hdf5_file,t, {
                    let var_name1 = strip(&vars.dvars[id1].0);
                    let var_name2 = strip(&vars.dvars[id2].0);
                    let w = vars.dvars[id1].1;
                    if w != 1 {
                        panic!("Fields considered by Anysotropy observable must be scalar.");
                    }
                    if w != vars.dvars[id2].1 {
                        panic!("Both fields in Anisotropy observable must have the same properties");
                    }

                    let ap = ReduceParam {
                        vect_dim: w,
                        dst_size: None,
                        window: None,
                    };

                    let mut dim: [usize;3] = vars.dim.into();
                    let d = dim.iter().fold(1, |a, i| a * i);
                    h.run_arg("anisotropy", D1(d), &[BufArg(&vars.dvars[id1].0, "x"),BufArg(&vars.dvars[id2].0, "y"),BufArg("tmp", "dst")])?;
                    h.run_algorithm(
                        "sum",
                        vars.dim,
                        &vars.dirs,
                        &["tmp", "tmp2", "sum"],
                        AlgorithmParam::Ref(&ap),
                    )?;

                    if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                        vars.dirs.iter().for_each(|d| dim[*d as usize] = 1);
                        let len = dim[0]*dim[1]*dim[2];
                        let num = 4;
                        let prm = MomentsParam{ num: num as _, vect_dim: w, packed: false };
                        h.run_algorithm("moments", D2(1,len), &[Y], &["sum","tmp","tmp2","tmp3"], Ref(&prm))?; // WARNING use "cumulants" as tmp buffer
                        let res = h.get_firsts("tmp3",num*w as usize)?.VF64();
                        if let Some(hdf5) = hdf5_file {
                            hdf5write!(hdf5, vars.parent, t, "anisotropy", &format!("{}-{}", var_name1, var_name2), m &res.into_iter().map(|m| vec![m]).collect::<Vec<_>>());
                            attr!(hdf5, vars.parent, t, "anisotropy", "noise_configurations", &len);
                        } else {
                            let moms = vvtos(&res, w as usize, venot);
                            write_all(&vars.parent, "anisotropy.txt", &format!("{:e}|{}|{}#{}\n", t, format!("{}-{}", var_name1, var_name2), len, moms));
                        }

                    } else {
                        // WARNING: considers that w is 1
                        let anisotropy = h.get_first("sum")?.F64();
                        if let Some(hdf5) = hdf5_file {
                            hdf5write!(hdf5, vars.parent, t, "anisotropy", &format!("{}-{}", var_name1, var_name2), &vec![anisotropy]);
                        } else {
                            write_all(&vars.parent, "anisotropy.txt", &format!("{:e}|{}|{}\n", t, format!("{}-{}", var_name1, var_name2), anisotropy));
                        }
                    }
                }}
            }
        }
    }
}
pub fn moments(a: Vec<Vec<Radial>>, num: usize) -> Vec<Vec<Radial>> {
    let nbsim = a.len();
    let len = a[0].len();
    let w = a[0][0].vals.len();
    let mut moms: Vec<Vec<Radial>> = vec![
        a[0].iter()
            .map(|r| Radial {
                pos: r.pos,
                vals: r.vals.iter().map(|_| 0.0).collect()
            })
            .collect();
        num
    ];
    for vs in a {
        for (i, v) in vs.iter().enumerate() {
            let mut tmp = (0..w).map(|_| 1.0).collect::<Vec<_>>();
            for p in 0..num {
                for j in 0..w {
                    tmp[j] *= v.vals[j];
                    moms[p][i].vals[j] += tmp[j];
                }
            }
        }
    }
    for i in 0..len {
        for j in 0..w {
            for n in 0..num {
                moms[n][i].vals[j] /= nbsim as f64;
            }
        }
    }
    moms
}
pub fn cumulants(a: Vec<Vec<Radial>>, num: usize) -> Vec<Vec<Radial>> {
    let len = a[0].len();
    let w = a[0][0].vals.len();
    let mut moms: Vec<Vec<Radial>> = moments(a, num);
    for i in 0..len {
        for j in 0..w {
            let c =
                moments_to_cumulants(&(0..num).map(|n| moms[n][i].vals[j]).collect::<Vec<_>>(), 1);
            for n in 0..num {
                moms[n][i].vals[j] = c[n];
            }
        }
    }
    moms
}
