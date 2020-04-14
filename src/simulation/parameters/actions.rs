use serde::{Deserialize,Serialize};
use crate::simulation::Vars;
use gpgpu::algorithms::moments_to_cumulants;
use std::io::Write;

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Moments,
    Correlation,
    StaticStructureFactor,
}
pub use Action::*;

pub type Callback = Box<dyn Fn(&mut gpgpu::Handler,&Vars,f64) -> gpgpu::Result<()>>;

fn write_all<'a>(file_name: &'a str, content: &'a str) {
    let write = |f,c: &str| std::fs::OpenOptions::new().create(true).append(true).open(f)?.write_all(c.as_bytes());
    if let Err(e) = write(file_name,content) {
        eprintln!("Could not write to file \"{}\"",file_name);
    }
}

impl Action {
    pub fn to_callback(&self) -> Callback {
        match self {
            Moments => Box::new(|h,vars,t| {
                h.run_algorithm("moments", vars.dim, &vars.dirs, &["u","tmp","sum","sumdst","moments"], None)?;
                let moments = h.get::<f64>("moments")?;
                let cumulants = moments_to_cumulants(&moments);
                write_all("moments.yaml", &format!("- {}:\n  moments: [{}]\n  cumulants: [{}]\n", t,
                        moments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                        cumulants.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            Correlation => Box::new(|h,vars,t| {
                h.run_algorithm("correlation", vars.dim, &vars.dirs, &["u","tmp"], None)?;
                let correlation = h.get::<f64>("tmp")?;
                write_all("correlation.yaml", &format!("- {}: [{}]\n", t,
                        correlation.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
            StaticStructureFactor => Box::new(|h,vars,t| {
                h.run_algorithm("FFT", vars.dim, &vars.dirs, &["u","tmpFFT","dstFFT"], None)?;
                let fft = h.get::<gpgpu::Double2>("dstFFT")?;
                write_all("static_structure_factor.yaml", &format!("- {:e}: [{}]\n", t,
                        fft.iter().map(|i| format!("[{},{}]",i[0],i[1])).collect::<Vec<_>>().join(",")
                ));

                Ok(())
            }),
        }
    }
}
