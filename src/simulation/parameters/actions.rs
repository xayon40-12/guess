use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::moments_to_cumulants;
use std::io::Write;
use gpgpu::Dim::*;

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Moments,
    Correlation,
    StaticStructureFactor,
}
pub use Action::*;

pub type Callback = Box<dyn Fn(&mut gpgpu::Handler,&Vars,f64) -> gpgpu::Result<()>>;

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
                let moments = h.get::<f64>("moments")?;
                let cumulants = moments_to_cumulants(&moments);
                write_all("moments.yaml", &format!("- t: {:e}\n  moments: [{}]\n  cumulants: [{}]\n", t,
                        moments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                        cumulants.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            Correlation => Box::new(|h,vars,t| {
                h.run_algorithm("correlation", vars.dim, &vars.dirs, &["u","tmp"], None)?;
                let correlation = h.get::<f64>("tmp")?;
                write_all("correlation.yaml", &format!("- t: {:e}\n  correlation: [{}]\n", t,
                        correlation.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            StaticStructureFactor => Box::new(|h,vars,t| {
                h.run("complex_from_real", D1(vars.len))?;
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["srcFFT","tmpFFT","dstFFT"], None)?;
                h.run("kc_sqrmod", D1(vars.len))?;
                let fft = h.get::<f64>("tmp")?;
                write_all("static_structure_factor.yaml", &format!("- t: {:e}\n  Sk: [{}]\n", t,
                        fft.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
        }
    }
}
