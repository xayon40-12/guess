use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::moments_to_cumulants;
use std::io::Write;
use gpgpu::Dim::*;
use gpgpu::descriptors::KernelArg::*;

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Moments,
    Correlation,
    StaticStructureFactor,
    DynamicStructureFactor,
}
pub use Action::*;

pub type Callback = Box<dyn FnMut(&mut gpgpu::Handler,&Vars,f64) -> gpgpu::Result<()>>;

fn write_all<'a>(file_name: &'a str, content: &'a str) {
    let write = |f,c: &str| std::fs::OpenOptions::new().create(true).append(true).open(&format!("target/{}",f))?.write_all(c.as_bytes());
    if let Err(e) = write(file_name,content) {
        eprintln!("Could not write to file \"{}\".\n{:?}",file_name,e);
    }
}

impl Action {
    pub fn to_callback(&self) -> Callback {
        match self {
            Moments => Box::new(|h,vars,t| {
                h.run_algorithm("moments", vars.dim, &vars.dirs, &["u","tmp","sum","sumdst","moments"], None)?;
                let moments = h.get("moments")?.VF64();
                let cumulants = moments_to_cumulants(&moments);
                write_all("moments.yaml", &format!("- t: {:e}\n  moments: [{}]\n  cumulants: [{}]\n", t,
                        moments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                        cumulants.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            Correlation => Box::new(|h,vars,t| {
                h.run_algorithm("correlation", vars.dim, &vars.dirs, &["u","tmp"], None)?;
                let correlation = h.get("tmp")?.VF64();
                write_all("correlation.yaml", &format!("- t: {:e}\n  correlation: [{}]\n", t,
                        correlation.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            StaticStructureFactor => Box::new(|h,vars,t| {
                h.run_arg("complex_from_real", D1(vars.len), &[BufArg("u","src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], None)?;
                h.run_arg("kc_sqrmod", D1(vars.len), &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;
                let fft = h.get("tmp")?.VF64();
                write_all("static_structure_factor.yaml", &format!("- t: {:e}\n  Sk: [{}]\n", t,
                        fft.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            DynamicStructureFactor => { let mut first = true; let mut start = "\n-"; Box::new(move |h,vars,t| {
                h.run_arg("complex_from_real", D1(vars.len), &[BufArg("u","src"),BufArg("srcFFT","dst")])?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], None)?;
                if first {
                    first = false;
                    h.copy("dstFFT","initFFT")?;
                }
                h.run_arg("kc_times", D1(vars.len), &[BufArg("initFFT","a"),BufArg("dstFFT","b"),BufArg("dstFFT","dst")])?;
                h.run_arg("kc_sqrmod", D1(vars.len), &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;
                let fft = h.get("tmp")?.VF64();
                write_all("dynamic_structure_factor.yaml", &format!("{} - t: {:e}\n    Sk: [{}]\n", &start, t,
                        fft.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));
                start = " ";

                Ok(())
            })},
        }
    }
}
