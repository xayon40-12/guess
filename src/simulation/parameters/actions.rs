use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::moments_to_cumulants;
use std::io::Write;
use gpgpu::Dim::*;
use gpgpu::descriptors::KernelArg::*;

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Moments,
    StaticStructureFactor,
    DynamicStructureFactor,
    RawData,
}
pub use Action::*;

pub type Callback = Box<dyn FnMut(&mut gpgpu::Handler,&Vars,f64) -> gpgpu::Result<()>>;

fn write_all<'a>(file_name: &'a str, content: &'a str) {
    let write = |f,c: &str| std::fs::OpenOptions::new().create(true).append(true).open(&format!("target/{}",f))?.write_all(c.as_bytes());
    if let Err(e) = write(file_name,content) {
        eprintln!("Could not write to file \"{}\".\n{:?}",file_name,e);
    }
}

impl Action { //WARNING these actions only work on scalar data yet (vectorial not supported)
    // To use vectorial data, the dim of the data must be known here
    pub fn to_callback(&self) -> Callback {
        match self {
            Moments => Box::new(|h,vars,t| {
                h.run_algorithm("moments", vars.dim, &vars.dim.all_dirs(), &[&vars.dvars[1],"tmp","sum","sumdst","moments"], None)?;
                let moments = h.get("moments")?.VF64();
                let cumulants = moments_to_cumulants(&moments);
                write_all("moments.yaml", &format!("- t: {:e}\n  moments: [{}]\n  cumulants: [{}]\n", t,
                        moments.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(","),
                        cumulants.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            StaticStructureFactor => Box::new(|h,vars,t| {
                h.run_arg("complex_from_real", D1(vars.len), &[BufArg(&vars.dvars[1],"src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], None)?;
                h.run_arg("kc_sqrmod", D1(vars.len), &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;
                let mut len = vars.len;
                let mut phy = vars.phy.clone();
                phy.iter_mut().for_each(|i| if *i == 0.0 { *i = 1.0 });
                let mut size = phy[0]*phy[1]*phy[2];
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    let mut dirs = vars.dim.all_dirs();
                    dirs.retain(|v| !vars.dirs.contains(&v));
                    h.run_algorithm("sum",dim.into(),&dirs,&["tmp","sum","tmp"],None)?;
                    dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    len = dim[0]*dim[1]*dim[2];
                    size /= (vars.len/len) as f64;
                }
                h.run_arg("ctimes", D1(len), &[BufArg("tmp","src"),Param("c",size.into()),BufArg("tmp","dst")])?;
                let fft = h.get_firsts("tmp",len)?.VF64();
                write_all("static_structure_factor.yaml", &format!("- t: {:e}\n  Sk: [{}]\n", t,
                        fft.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            DynamicStructureFactor => { let mut first = true; let mut start = "\n-"; Box::new(move |h,vars,t| {
                h.run_arg("complex_from_real", D1(vars.len), &[BufArg(&vars.dvars[1],"src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], None)?;
                if first {
                    first = false;
                    h.copy("dstFFT","initFFT")?;
                }
                h.run_arg("kc_times_conj", D1(vars.len), &[BufArg("initFFT","a"),BufArg("dstFFT","b"),BufArg("dstFFT","dst")])?;
                let mut len = vars.len;
                let mut phy = vars.phy.clone();
                phy.iter_mut().for_each(|i| if *i == 0.0 { *i = 1.0 });
                let mut size = phy[0]*phy[1]*phy[2];
                if vars.dim.len() > 1 && vars.dirs.len() != vars.dim.len() {
                    let mut dim: [usize;3] = vars.dim.into();
                    dim[0] *= 2;//TODO remove and use vectorial sum instead
                    let mut dirs = vars.dim.all_dirs();
                    dirs.retain(|v| !vars.dirs.contains(&v));
                    //TODO if X is in dirs, the sum must be a vectorial sum of dim 2
                    h.run_algorithm("sum",dim.into(),&dirs,&["dstFFT","tmpFFT","dstFFT"],None)?;
                    dirs.iter().for_each(|d| dim[*d as usize] = 1);
                    if dim[0] != 1 { dim[0] /= 2; }//TODO remove and use vectorial sum instead
                    len = dim[0]*dim[1]*dim[2];
                    size /= (vars.len/len) as f64;
                }
                h.run_arg("ctimes", D1(len*2), &[BufArg("dstFFT","src"),Param("c",size.into()),BufArg("dstFFT","dst")])?;
                let fft = h.get_firsts("dstFFT",len)?.VF64_2();
                write_all("dynamic_structure_factor.yaml", &format!("{} - t: {:e}\n    Sk: [{}]\n", &start, t,
                        fft.iter().flatten().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));
                start = " ";

                Ok(())
            })},
            RawData => Box::new(|h,vars,t| {
                let raw = h.get(&vars.dvars[1])?.VF64();
                write_all("raw.yaml", &format!("- t: {:e}\n  raw: [{}]\n", t,
                        raw.iter().map(|i| format!("{:e}", i)).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
        }
    }
}
