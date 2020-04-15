use gpgpu::{Handler,handler::HandlerBuilder};
use gpgpu::data_file::Format;
use gpgpu::{Dim::{self,*},DimDir};
use gpgpu::descriptors::{Type::*,BufferConstructor::*,KernelArg::*};

use std::time::{SystemTime, UNIX_EPOCH};

pub mod parameters;
pub use parameters::{actions::Callback,activations::ActivationCallback,Param};

pub struct Vars {
    max_count: usize,
    dim: Dim,
    dirs: Vec<DimDir>,
    dt: f64,
    len: usize,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback,Callback)>,
    vars: Vars,
}

impl Simulation {
    pub fn from_param<'a>(file_name: &'a str) -> gpgpu::Result<Self> {
        let param: Param = serde_yaml::from_str(&std::fs::read_to_string(file_name).expect(&format!("Could not find parameter file \"{}\".", file_name))).unwrap();
        println!("param:\n{:?}", &param);

        let mut handler = Handler::builder()?;

        for f in &param.data_files {
            let name = if let Some(i) = f.find('.') { &f[..i] } else { f };
            let name = if let Some(i) = name.rfind('/') { &name[i+1..] } else { name };
            handler = handler.load_data(name,Format::Column(&std::fs::read_to_string(f).expect(&format!("Could not find data file \"{}\".", f))),false,None); //TODO autodetect format from file extension
        }

        let (handler,vars) = extract_symbols(handler, &param)?;
        let callbacks = param.actions.iter().map(|(c,a)| (a.to_activation(vars.dt),c.to_callback())).collect();

        Ok(Simulation { handler, callbacks, vars })
    }

    pub fn run(&mut self) -> gpgpu::Result<()> {
        let Vars {max_count, dim: _, dirs: _, dt, len} = self.vars;
        for c in 0..max_count {
            let t = c as f64*dt;

            self.handler.run("noise", D1(len))?;

            self.handler.copy::<f64>("noise","u")?;
            //self.handler.run_arg("simu", dim, &[Param("t", F64(t))])?;

            for (activator,callback) in &self.callbacks {
                if activator(c) {
                    callback(&mut self.handler, &self.vars, t)?;
                }
            }
        }

        Ok(())
    }
}

fn extract_symbols<'a>(mut h: HandlerBuilder<'a>, param: &'a Param) -> gpgpu::Result<(Handler,Vars)> {

    let dt = 0.1;
    let dim = param.config.dim;
    let dirs = param.config.dirs.clone();
    let max_count = param.config.max.convert(dt);

    let dims: [usize;3] = dim.into();
    let mut sumdims = dims.clone();
    dirs.iter().for_each(|d| sumdims[*d as usize] = 1);
    let len = dims[0]*dims[1]*dims[2];
    let lensum = sumdims[0]*sumdims[1]*sumdims[2];

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

    h = h.add_buffer("srcnoise", Len(U64(time),len));
    h = h.add_buffer("noise", Len(F64(0.0),len));
    h = h.add_buffer("u", Len(F64(0.0), len));
    h = h.add_buffer("tmp", Len(F64(0.0), len));
    h = h.add_buffer("sum", Len(F64(0.0), len));
    h = h.add_buffer("sumdst", Len(F64(0.0), lensum));
    h = h.add_buffer("moments", Len(F64(0.0), 4));
    h = h.add_buffer("srcFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.add_buffer("tmpFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.add_buffer("dstFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.load_all_algorithms();
    h = h.load_kernel_named("philox4x32_10_normal","noise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");

    let mut h = h.build()?;
    h.set_arg("noise", &[BufArg("srcnoise","src"),BufArg("noise","dst")])?;
    h.set_arg("complex_from_real", &[BufArg("u","src"),BufArg("srcFFT","dst")])?;// WARNING these kernel args must not be changed
    h.set_arg("kc_sqrmod", &[BufArg("dstFFT","src"),BufArg("tmp","dst")])?;// WARNING these kernel args must not be changed

    Ok((h,Vars { max_count, dim, dirs, dt, len }))
}
