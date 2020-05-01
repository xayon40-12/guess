use gpgpu::{Handler,handler::HandlerBuilder};
use gpgpu::data_file::Format;
use gpgpu::{Dim::{self,*},DimDir};
use gpgpu::descriptors::{Types::*,ConstructorTypes::*,BufferConstructor::*,KernelArg::*,SFunctionConstructor::*};

#[cfg(debug_assertions)]
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod parameters;
pub use parameters::{Callback,ActivationCallback,Param,Integrator,SymbolsTypes};
use parameters::Noises::{self,*};

pub struct Vars {
    pub t_max: f64,
    pub dim: Dim,
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub dvars: Vec<String>,
    pub noises: Vec<Noises>,
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

        extract_symbols(handler, param)
    }

    pub fn run(&mut self) -> gpgpu::Result<()> {
        let Vars {t_max, dim, dirs: _, len, ref dvars, ref noises } = self.vars;
        let noise_dim = |dim: &Option<usize>| D1(len*if let Some(d) = dim { *d } else { 1 }/2);//WARNING must divide by 2 because random number are computed 2 at a time
        let dvars = dvars.iter().map(|i| &i[..]).collect::<Vec<_>>();

        let mut t = 0.0;
        for (activator,callback) in &mut self.callbacks {
            if activator(t) {
                callback(&mut self.handler, &self.vars, t)?;
            }
        }
        #[cfg(debug_assertions)]
        let (mut pred,int) = (t,t_max/100.0);
        while t<t_max {
            #[cfg(debug_assertions)]
            if t>=pred+int {
                pred = t;
                print!(" {}%\r",f64::trunc(t/int)); std::io::stdout().lock().flush().unwrap(); 
            }
            for noise in noises {
                match noise {
                    Uniform{name,dim} => self.handler.run_arg("unifnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                    Normal{name,dim} => self.handler.run_arg("normnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                }
            }

            t = *self.handler.run_algorithm("integrator",dim,&[],&dvars[..],Some(&(t,vec![("t".to_string(),F64(t))])))?.unwrap().downcast_ref::<f64>().unwrap();

            for (activator,callback) in &mut self.callbacks {
                if activator(t) {
                    callback(&mut self.handler, &self.vars, t)?;
                }
            }
        }

        Ok(())
    }
}

fn extract_symbols(mut h: HandlerBuilder, mut param: Param) -> gpgpu::Result<Simulation> {

    let (dims,phy) = param.config.dim.into();
    let dim: Dim = dims.into();
    let dirs = param.config.dirs.clone();
    let t_max = param.config.t_max;

    let mut sumdims = dims.clone();
    dirs.iter().for_each(|d| sumdims[*d as usize] = 1);
    let len = dims[0]*dims[1]*dims[2];
    let lensum = sumdims[0]*sumdims[1]*sumdims[2];

    //TODO verify that there is no link between two noise that start whith an initial condition
    //that differ of 1.
    let mut time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

    h = h.add_buffer("tmp", Len(F64(0.0), len));
    h = h.add_buffer("sum", Len(F64(0.0), len));
    h = h.add_buffer("sumdst", Len(F64(0.0), lensum));
    h = h.add_buffer("moments", Len(F64(0.0), 4));
    h = h.add_buffer("srcFFT", Len(F64_2([0.0,0.0]), len));
    h = h.add_buffer("tmpFFT", Len(F64_2([0.0,0.0]), len));
    h = h.add_buffer("dstFFT", Len(F64_2([0.0,0.0]), len));
    h = h.add_buffer("initFFT", Len(F64_2([0.0,0.0]), len));
    h = h.load_algorithm("moments");
    h = h.load_algorithm("correlation");
    h = h.load_kernel_named("philox4x32_10_unit","unifnoise");
    h = h.load_kernel_named("philox4x32_10_normal","normnoise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");
    h = h.load_kernel("kc_times");

    let dvars = {
        use SymbolsTypes::*;
        use gpgpu::integrators::{create_euler_pde};
        use gpgpu::functions::SFunction;
        use std::collections::{HashSet,HashMap};
        use regex::Regex;
        let mut consts_names = HashSet::new();
        let mut consts: Vec<Box<dyn Fn(String) -> String>> = vec![];
        for i in 0..3 {
            if phy[i] != 0.0 {
                let d = phy[i]/dims[i] as f64;
                let symb = ["dx","dy","dz"][i];
                consts_names.insert(symb.to_string());
                let re = Regex::new(&format!(r"\b{}\b",symb)).expect(&format!("Could not build regex out of {:?}.", symb));
                let val = format!("{:e}",d);
                consts.push(Box::new(move |s| re.replace_all(&s,&val[..]).to_string()));
                let symb = ["ivdx","ivdy","ivdz"][i];
                consts_names.insert(symb.to_string());
                let re = Regex::new(&format!(r"\b{}\b",symb)).expect(&format!("Could not build regex out of {:?}.", symb));
                let val = format!("{:e}",1.0/d);
                consts.push(Box::new(move |s| re.replace_all(&s,&val[..]).to_string()));
            }
        }
        match &param.integrator {
            Integrator::Euler{dt} => {
                let symb = "dt";
                let re = Regex::new(&format!(r"\b{}\b",symb)).expect(&format!("Could not build regex out of {:?}.", symb));
                let val = format!("{:e}",dt);
                consts.push(Box::new(move |s| re.replace_all(&s,&val[..]).to_string()));
                let symb = "ivdt";
                let re = Regex::new(&format!(r"\b{}\b",symb)).expect(&format!("Could not build regex out of {:?}.", symb));
                let val = format!("{:e}",1.0/dt);
                consts.push(Box::new(move |s| re.replace_all(&s,&val[..]).to_string()));
            },
        }
        for symb in &param.symbols {
            match symb {
                Constant{name,value} => {
                    if !consts_names.insert(name.clone()) { panic!("Each constants must have a different name, repeated name: \"{}\"", name) }
                    else {
                        let re = Regex::new(&format!(r"\b{}\b",&name)).expect(&format!("Could not build regex out of {:?}.", symb));
                        let val = format!("{:e}",value);
                        consts.push(Box::new(move |s| re.replace_all(&s,&val[..]).to_string()));
                    }
                },
                _ => {}
            }
        }
        let replace = |s: String| {
            let mut s = s;
            for r in &consts {
                s = r(s);
            }
            s
        };
        let mut pdes_dvars = None;
        let mut noises_names = vec![];
        let mut noises_names_set = HashSet::new();
        for noise in &param.noises {
            match noise {
                Uniform{name,..} | Normal{name,..} => {
                    noises_names.push(name.to_string());
                    if !noises_names_set.insert(name) { panic!("Each noises must have a different name.") }
                },
            }
        }
        for symb in param.symbols {
            match symb {
                Constant{..} => {},
                Function{name,args,indx_args,src} => {
                    let mut args = args.into_iter().map(|a| FCParam(a,CF64)).collect::<Vec<_>>();
                    if let Some(ia) = indx_args { 
                        for a in ia {
                            args.push(FCGlobalPtr(a,CF64));
                        }
                    }

                    h = h.create_function(SFunction {
                        name,
                        args,
                        src: format!("return {};", replace(src)),
                        ret_type: Some(CF64),
                        needed: vec![],
                    });

                },
                PDEs(mut pdes) => {
                    pdes_dvars = Some(pdes.iter().map(|i| i.dvar.clone()).collect::<Vec<_>>());
                    pdes.iter_mut().for_each(|i| i.expr.iter_mut().for_each(|e| *e=replace(e.clone())));
                    match &param.integrator {
                        Integrator::Euler{dt} => h = h.create_algorithm(create_euler_pde("integrator",*dt,pdes,Some(noises_names.clone()),vec![("t".into(),CF64)])),
            //TODO add all noises
                    }
                },
            }
        }
        let dvars = if let Some(mut dvars) = pdes_dvars {
            dvars.iter_mut().for_each(|i| *i = format!("dvar_{}",i));
            dvars.insert(0,"dst".to_string());
            for dvar in &dvars {
                h = h.add_buffer(dvar,Len(F64(0.0),len));
            }
            for noise in &param.noises {
                match noise {
                    Uniform{name,dim} | Normal{name,dim} => {
                        let l = len*if let Some(d) = dim { if *d==0 { panic!("dim of noise \"{}\" must be different of 0.",name) } else { *d } } else { 1 };
                        h = h.add_buffer(&format!("src{}",name),Len(U64(time),l));
                        time += 1;
                        h = h.add_buffer(name,Len(F64(0.0),l));
                        dvars.push(name.clone());
                    },
                }
            }
            dvars
        } else {
            panic!("PDEs must be given.")
        };
        if let Some(file) = param.initial_conditions_file {
            let init: HashMap<String,Vec<f64>> = serde_yaml::from_str(&std::fs::read_to_string(&file).expect(&format!("Could not find initial conditions file \"{}\".", &file))).unwrap();


        }
        dvars
    };

    for noise in &mut param.noises {
        match noise {
            Uniform{name,..} => *name = format!("src{}",name),
            Normal{name,..} => *name = format!("src{}",name),
        }
    }

    let handler = h.build()?;

    let vars = Vars { t_max, dim, dirs, len, dvars, noises: param.noises };
    let callbacks = param.actions.into_iter().map(|(c,a)| (a.to_activation(),c.to_callback())).collect();

    Ok(Simulation { handler, callbacks, vars })
}
