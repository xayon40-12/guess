use gpgpu::{Handler,handler::HandlerBuilder};
use gpgpu::data_file::Format;
use gpgpu::{Dim::{self,*},DimDir};
use gpgpu::descriptors::{Types::*,BufferConstructor::*,KernelArg::*};

use std::time::{SystemTime, UNIX_EPOCH};

pub mod parameters;
pub use parameters::{Callback,ActivationCallback,Param,Integrator};

pub struct DataBuffer {
    name: String,
    vector_dim: usize,
}

pub struct Vars {
    pub max: f64,
    pub dim: Dim,
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub data_buffers: Vec<DataBuffer>,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback,Callback)>,
    vars: Vars,
    integrators: Vec<Integrator>,
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

        extract_symbols(handler, param)
    }

    pub fn run(&mut self) -> gpgpu::Result<()> {
        let Vars {max, dim, dirs: _, len, data_buffers: _} = self.vars;
        let noise_dim = D1(len*dim.len());

        let dt = 0.1;//TODO remove
        let mut t = 0.0;
        while t<max {
            self.handler.run("noise", noise_dim)?;

            self.handler.copy("noise","u")?;
            //self.handler.run_arg("simu", dim, &[Param("t", F64(t))])?;

            t += dt;//TODO make t evolve with the integration algorithm
            for (activator,callback) in &mut self.callbacks {
                if activator(t) {
                    callback(&mut self.handler, &self.vars, t)?;
                }
            }
        }

        Ok(())
    }
}

fn extract_symbols(mut h: HandlerBuilder, param: Param) -> gpgpu::Result<Simulation> {

    let dim = param.config.dim;
    let dirs = param.config.dirs.clone();
    let max = param.config.max;

    let dims: [usize;3] = dim.into();
    let mut sumdims = dims.clone();
    dirs.iter().for_each(|d| sumdims[*d as usize] = 1);
    let len = dims[0]*dims[1]*dims[2];
    let lensum = sumdims[0]*sumdims[1]*sumdims[2];

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

    let noise_len = len*dim.len();

    h = h.add_buffer("srcnoise", Len(U64(time),noise_len));
    h = h.add_buffer("noise", Len(F64(0.0),noise_len));
    h = h.add_buffer("tmp", Len(F64(0.0), len));
    h = h.add_buffer("sum", Len(F64(0.0), len));
    h = h.add_buffer("sumdst", Len(F64(0.0), lensum));
    h = h.add_buffer("moments", Len(F64(0.0), 4));
    h = h.add_buffer("srcFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.add_buffer("tmpFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.add_buffer("dstFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.add_buffer("initFFT", Len(F64_2([0.0,0.0].into()), len));
    h = h.load_all_algorithms();
    h = h.load_kernel_named("philox4x32_10_normal","noise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");
    h = h.load_kernel("kc_times");

    let data_buffers = vec![DataBuffer{name:"u".into(),vector_dim:1}];//TODO load the names from the parameters
    for db in &data_buffers {
        h = h.add_buffer(&db.name, Len(F64(0.0), len*db.vector_dim));
    }

    let mut handler = h.build()?;
    handler.set_arg("noise", &[BufArg("srcnoise","src"),BufArg("noise","dst")])?;

    let vars = Vars { max, dim, dirs, len, data_buffers };
    let callbacks = param.actions.into_iter().map(|(c,a)| (a.to_activation(),c.to_callback())).collect();
    let integrators = param.integrators;

    Ok(Simulation { handler, callbacks, vars, integrators })
}
