use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::{moments_to_cumulants,AlgorithmParam::*,MomentsParam};
use std::io::Write;
use gpgpu::Dim::*;
use gpgpu::descriptors::KernelArg::*;
use std::collections::HashMap;

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum Action {
    Moments(Vec<String>),
    StaticStructureFactor(Vec<String>),
    DynamicStructureFactor(Vec<String>),
    Correlation(Vec<String>),
    RawData(Vec<String>),
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

impl Action { //WARNING these actions only work on scalar data yet (vectorial not supported)
    // To use vectorial data, the dim of the data must be known here
    pub fn to_callback(&self, name_to_index: &HashMap<String,usize>, num_pdes: usize) -> Callback {
        match self {
            Moments(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                h.run_algorithm("moments", vars.dim, &vars.dim.all_dirs(), &[&vars.dvars[id].0,"tmp","sum","sumdst","moments"], Ref(&(4u32,w)))?;
                let moments = h.get_firsts("moments",4*w as usize)?.VF64();
                let cumulants = moments_to_cumulants(&moments, w as _);
                write_all(&vars.parent, "moments.yaml", &format!("{}  {}:\n    moments: [{}]\n    cumulants: [{}]\n",if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        moments.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","),
                        cumulants.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));
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
                    let mut dim: [usize;3] = vars.dim.into();
                    let mut dirs = vars.dim.all_dirs();
                    dirs.retain(|v| !vars.dirs.contains(&v));
                    let prm = MomentsParam{ num: 2, vect_dim: w, packed: false };
                    h.run_algorithm("moments", dim.into(),&dirs, &["tmp","tmp","sum","sumdst"], Ref(&prm))?;
                    dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    let len = dim[0]*dim[1]*dim[2];
                    let mom = h.get_firsts("sumdst",len*2*w as usize)?.VF64().chunks(len*w as usize)
                        .map(|c| c.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","))
                        .collect::<Vec<String>>();
                    write_all(&vars.parent, "static_structure_factor.yaml", &format!("{}  {}:\n    SF: [{}]\n    var_SF: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        mom[0], mom[1]
                    ));
                } else {
                    let mom = h.get_firsts("tmp",len*w as usize)?.VF64();
                    write_all(&vars.parent, "static_structure_factor.yaml", &format!("{}  {}:\n    SF: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        mom.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                    ));
                }
            }},
            DynamicStructureFactor(names) =>  { let mut first = true; let mut start = "\n-"; gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let mut len = vars.len;
                h.run_arg("complex_from_real", D1(len*w as usize), &[BufArg(&vars.dvars[id].0,"src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], Ref(&w))?;
                if first {
                    first = false;
                    h.copy("dstFFT","initFFT")?;
                }
                h.run_arg("kc_times_conj", D1(len*w as usize), &[BufArg("initFFT","a"),BufArg("dstFFT","b"),BufArg("dstFFT","dst")])?;
                let mut phy = vars.phy.clone();
                phy.iter_mut().for_each(|i| if *i == 0.0 { *i = 1.0 });
                let mut size = phy[0]*phy[1]*phy[2];
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    let mut dirs = vars.dim.all_dirs();
                    dirs.retain(|v| !vars.dirs.contains(&v));
                    h.run_algorithm("sum",dim.into(),&dirs,&["dstFFT","tmpFFT","dstFFT"],Ref(&(w*2u32)))?;
                    dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    len = dim[0]*dim[1]*dim[2];
                    size /= (vars.len/len) as f64;
                }
                h.run_arg("ctimes", D1(len*w as usize*2), &[BufArg("dstFFT","src"),Param("c",size.into()),BufArg("dstFFT","dst")])?;
                let fft = h.get_firsts("dstFFT",len*w as usize)?.VF64_2();
                write_all(&vars.parent, "dynamic_structure_factor.yaml", &format!("{}    {}:\n      DSF: [{}]\n", if head { format!("{} - t: {:e}\n", &start, t) } else { "".into() }, strip(&vars.dvars[id].0),
                        fft.iter().flatten().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));
                start = " ";
            }}},
            Correlation(names) => gen!{names,id,head,name_to_index,num_pdes,h,vars,t, {
                let w = vars.dvars[id].1;
                let mut len = vars.len;
                h.run_algorithm("correlation",vars.dim,&vars.dirs,&[&vars.dvars[id].0,"tmp"],Ref(&w))?;
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    let mut dirs = vars.dim.all_dirs();
                    dirs.retain(|v| !vars.dirs.contains(&v));
                    h.run_algorithm("sum",dim.into(),&dirs,&["tmp","sum","tmp"],Ref(&w))?;
                    dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    len = dim[0]*dim[1]*dim[2];
                    let size = len as f64/vars.len as f64;
                    h.run_arg("ctimes", D1(len*w as usize), &[BufArg("tmp","src"),Param("c",size.into()),BufArg("tmp","dst")])?;
                }
                let correlation = h.get_firsts("tmp",len*w as usize)?.VF64();
                write_all(&vars.parent, "correlation.yaml", &format!("{}  {}:\n    correlation: [{}]\n", if head { format!("- t: {:e}\n", t) } else { "".into() }, strip(&vars.dvars[id].0),
                        correlation.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));
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
