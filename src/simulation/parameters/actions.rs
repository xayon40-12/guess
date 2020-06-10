use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::{moments_to_cumulants,AlgorithmParam::*,MomentsParam,ReduceParam,Window};
use std::io::Write;
use gpgpu::{Dim::*,DimDir::*};
use gpgpu::descriptors::KernelArg::*;
use std::collections::HashMap;

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum Action {
    Moments(Vec<String>),
    StaticStructureFactor(Vec<String>),
    DynamicStructureFactor(Vec<String>),
    Correlation(Vec<String>),
    RawData(Vec<String>),
    Window(Vec<String>,Vec<f64>), // buffers, fm apperture
}
pub use Action::*;

pub type Callback = Box<dyn FnMut(&mut gpgpu::Handler,&Vars,usize,f64) -> gpgpu::Result<()>>;

fn write_all<'a>(parent: &'a str, file_name: &'a str, content: &'a str) {
    let write = |f,c: &str| std::fs::OpenOptions::new().create(true).append(true).open(&format!("{}/{}",parent,f))?.write_all(c.as_bytes());
    if let Err(e) = write(file_name,content) {
        eprintln!("Could not write to file \"{}\".\n{:?}",file_name,e);
    }
}

macro_rules! gen {
    ($names:ident, $id:ident, $head:ident, $name_to_index:ident, $num_pdes:ident, $h:ident, $vars:ident, $t:ident, $body:tt) => {{
        let idx = $names.iter().map(|n| $name_to_index[n]).collect::<Vec<_>>();
        Box::new(move |$h,$vars,swap,$t| {
            let mut $head = true;
            for $id in idx.iter().map(|&i| if i<2*$num_pdes { i+swap } else { i }) {
                $body
                $head = false;
            }

            Ok(())
        })
    }};
}

fn strip<'a>(s: &'a str) -> String {
    s.replace("dvar_","").replace("swap_","").into()
}

fn moms(w: u32, vars: &Vars, bufs: &[&'static str], h: &mut gpgpu::Handler, complex: bool) -> gpgpu::Result<Vec<String>> {
    let mut dim: [usize;3] = vars.dim.into();
    let mut dirs = vars.dim.all_dirs();
    dirs.retain(|v| !vars.dirs.contains(&v));
    let mul = if complex { 2 } else { 1 };
    let prm = MomentsParam{ num: 2, vect_dim: w*mul, packed: false };
    h.run_algorithm("moments", dim.into(),&dirs, bufs, Ref(&prm))?;
    dirs.iter().for_each(|d| dim[*d as usize] = 1);
    let len = dim[0]*dim[1]*dim[2];
    h.run_arg("to_var",D1(len*w as usize*mul as usize),&[BufArg(bufs[3],"src")])?;
    let res = h.get_firsts(bufs[3],len*2*w as usize)?;
    let res = if complex {
        unsafe { 
            let mut res: Vec<f64> = std::mem::transmute(res.VF64_2());
            res.set_len(2*len*2*w as usize);
            res
        }
    } else {
        res.VF64()
    };
    Ok(res.chunks(len*w as usize*mul as usize)
        .map(|c| c.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","))
        .collect::<Vec<String>>())
}

impl Action { //WARNING these actions only work on scalar data yet (vectorial not supported)
    // To use vectorial data, the dim of the data must be known here
    pub fn to_callback(&self, name_to_index: &HashMap<String,usize>, num_pdes: usize) -> Callback {
        match self {
            Window(names,appertures) => {
                let appertures = appertures.clone();
                let mut current = String::new();
                gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                    if head { current = String::new(); }
                    let dim: [usize; 3] = vars.dim.into();
                    let windows: Vec<(Vec<Window>,usize,f64)> = appertures.iter().map(|app| {
                        let wins: Vec<Window> = vars.dirs.iter().map(|d| {
                            let dim = dim[*d as usize];
                            let mut len = (app*dim as f64) as _;
                            if len == 0 { len = 1 };
                            Window{ offset: (dim-len)/2, len}
                        }).collect();
                        let tot = wins.iter().fold(1, |a,w| a*w.len );
                        (wins,tot,*app)
                    }).collect();
                    let w = vars.dvars[id].1;
                    let num = 4;
                    let mut moments_app = String::new();
                    let mut cumulants_app = String::new();
                    for (window,tot,app) in windows {
                        let prm = ReduceParam{ vect_dim: w, dst_size: None, window: Some(window) };
                        h.run_algorithm("sum", vars.dim, &vars.dirs, &[&vars.dvars[id].0,"tmp","sum"], Ref(&prm))?;
                        let mut dim: [usize;3] = vars.dim.into();
                        vars.dirs.iter().for_each(|d| dim[*d as usize] = 1);
                        let len = dim[0]*dim[1]*dim[2];
                        h.run_arg("ctimes",D1(len*w as usize),&[BufArg("sum","src"),BufArg("sum","dst"),Param("c",(1.0/tot as f64).into())])?;
                        if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                            let prm = MomentsParam{ num: num as _, vect_dim: w, packed: true };
                            h.run_algorithm("moments", D1(len), &[X], &["sum","sum","tmp","summoments"], Ref(&prm))?;
                            let moments = h.get_firsts("summoments",num*w as usize)?.VF64();
                            let cumulants = moments_to_cumulants(&moments, w as _);
                            moments_app = format!("{}      {}: [{}]\n", moments_app, app, moments.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","));
                            cumulants_app = format!("{}      {}: [{}]\n", cumulants_app, app, cumulants.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","));
                        } else {
                            let win = h.get_firsts("sum",w as _)?.VF64();
                            let name = strip(&vars.dvars[id].0);
                            write_all(&vars.parent, "window.yaml", &format!("{}  {}\n      win: [{}]\n",
                                    if head { format!("- t: {:e}\n", t) } else { "".into() }, 
                                    if name != current { format!("{}:\n    {}:", name, app) } else { format!("  {}:", app) },
                            win.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","),
                            ));
                            head = false;
                            current = strip(&vars.dvars[id].0);
                        }
                    }
                    if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                        let name = strip(&vars.dvars[id].0);
                        write_all(&vars.parent, "window.yaml", &format!("{}  {}:\n    moments:\n{}    cumulants:\n{}",
                                if head { format!("- t: {:e}\n", t) } else { "".into() },
                                name,
                                moments_app,
                                cumulants_app
                        ));
                    }
                }}},
            Moments(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let num = 4;
                let prm = MomentsParam{ num: num as _, vect_dim: w, packed: true };
                h.run_algorithm("moments", vars.dim, &vars.dirs, &[&vars.dvars[id].0,"tmp","sum","moments"], Ref(&prm))?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    vars.dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    let len = dim[0]*dim[1]*dim[2];
                    let prm = MomentsParam{ num: 2, vect_dim: w, packed: false };
                    h.run_arg("moments_to_cumulants", D1(len), &[Buffer("moments"),Buffer("cumulants"),Param("vect_dim",w.into()),Param("num",(num as u32).into())])?;
                    h.run_algorithm("moments", D2(num,len), &[Y], &["moments","moments","summoments","sumdst"], Ref(&prm))?;
                    h.run_arg("to_var",D1(num*w as usize),&[BufArg("sumdst","src")])?;
                    let res = h.get_firsts("sumdst",num*2*w as usize)?.VF64();
                    let overall_cumulants = moments_to_cumulants(&res[0..(num*w as usize)],w as _).iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",");
                    let moms = res.chunks(num*w as usize)
                        .map(|c| c.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","))
                        .collect::<Vec<String>>();
                    h.run_algorithm("moments", D2(num,len), &[Y], &["cumulants","cumulants","summoments","sumdst"], Ref(&prm))?;
                    h.run_arg("to_var",D1(num*w as usize),&[BufArg("sumdst","src")])?;
                    let res = h.get_firsts("sumdst",num*2*w as usize)?.VF64();
                    let momsc = res.chunks(num*w as usize)
                        .map(|c| c.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","))
                        .collect::<Vec<String>>();
                    write_all(&vars.parent, "moments.yaml", &format!("{}  {}:\n    moments: [{}]\n    sigma_moments: [{}]\n    overall_cumulants: [{}]\n    cumulants: [{}]\n    sigma_cumulants: [{}]\n",if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moms[0],moms[1],overall_cumulants,momsc[0],momsc[1]
                    ));
                } else {
                    let moments = h.get_firsts("moments",num*w as usize)?.VF64();
                    let cumulants = moments_to_cumulants(&moments, w as _);
                    write_all(&vars.parent, "moments.yaml", &format!("{}  {}:\n    moments: [{}]\n    cumulants: [{}]\n",if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moments.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","),
                    cumulants.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                    ));
                }
            }},
            StaticStructureFactor(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let len = vars.len;
                h.run_arg("complex_from_real", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], Ref(&w))?;
                h.run_arg("kc_sqrmod", D1(len*w as usize), &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;
                let phy = vars.phy.iter().fold(1.0, |a,i| if *i == 0.0 { a } else { i*a });
                h.run_arg("ctimes", D1(len*w as usize), &[BufArg("tmp","src"),Param("c",phy.into()),BufArg("tmp","dst")])?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let moms = moms(w,&vars,&["tmp","tmp","sum","sumdst"],h,false)?;
                    write_all(&vars.parent, "static_structure_factor.yaml", &format!("{}  {}:\n    SF: [{}]\n    sigma_SF: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        moms[0], moms[1]
                    ));
                } else {
                    let moms = h.get_firsts("tmp",len*w as usize)?.VF64();
                    write_all(&vars.parent, "static_structure_factor.yaml", &format!("{}  {}:\n    SF: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        moms.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                    ));
                }
            }},
            DynamicStructureFactor(names) =>  { let mut first = true; let mut start = "-"; gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let len = vars.len;
                h.run_arg("complex_from_real", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], Ref(&w))?;
                if first {
                    first = false;
                    h.copy("dstFFT","initFFT")?;
                }
                h.run_arg("kc_times_conj", D1(len*w as usize), &[BufArg("initFFT","a"),BufArg("dstFFT","b"),BufArg("dstFFT","dst")])?;
                let phy = vars.phy.iter().fold(1.0, |a,i| if *i == 0.0 { a } else { i*a });
                h.run_arg("ctimes", D1(len*w as usize*2), &[BufArg("dstFFT","src"),Param("c",phy.into()),BufArg("dstFFT","dst")])?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let moms = moms(w,&vars,&["dstFFT","dstFFT","srcFFT","tmpFFT"],h,true)?;
                    write_all(&vars.parent, "dynamic_structure_factor.yaml", &format!("{}    {}:\n      DSF: [{}]\n      sigma_DSF: [{}]\n", if head { format!("{} - t: {:e}\n", &start, t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moms[0],moms[1]
                    ));
                } else {
                    let moms = h.get_firsts("dstFFT",len*w as usize)?.VF64_2();
                    write_all(&vars.parent, "dynamic_structure_factor.yaml", &format!("{}    {}:\n      DSF: [{}]\n", if head { format!("{} - t: {:e}\n", &start, t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moms.iter().flatten().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                    ));
                }
                start = " ";
            }}},
            Correlation(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let len = vars.len;
                h.run_algorithm("correlation",vars.dim,&vars.dirs,&[&vars.dvars[id].0,"tmp"],Ref(&w))?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let moms = moms(w,&vars,&["tmp","tmp","sum","sumdst"],h,false)?;
                    write_all(&vars.parent, "correlation.yaml", &format!("{}  {}:\n    correlation: [{}]\n    sigma_correlation: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moms[0],moms[1]
                    ));
                } else {
                    let moms = h.get_firsts("tmp",len*w as usize)?.VF64();
                    write_all(&vars.parent, "correlation.yaml", &format!("{}  {}:\n    correlation: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                    moms.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                    ));
                }
            }},
            RawData(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let len = vars.len;
                let raw = h.get_firsts(&vars.dvars[id].0,len*w as usize)?.VF64();
                write_all(&vars.parent, "raw.yaml", &format!("{}  {}:\n    raw: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        raw.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));
            }},
        }
    }
}
