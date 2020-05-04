use gpgpu::{Handler,handler::HandlerBuilder};
use gpgpu::data_file::Format;
use gpgpu::{Dim::{self,*},DimDir};
use gpgpu::descriptors::{Types::*,ConstructorTypes::*,BufferConstructor::*,KernelArg::*,SFunctionConstructor::*,KernelConstructor::*,SKernelConstructor};
use gpgpu::integrators::{create_euler_pde,SPDE};
use gpgpu::kernels::SKernel;
use gpgpu::functions::SFunction;

#[cfg(debug_assertions)]
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::{HashSet,HashMap};

pub mod parameters;
pub use parameters::{Callback,ActivationCallback,Param,Integrator,SymbolsTypes::{self,*},symbols::PrmType::{self,*}};
use parameters::Noises::{self,*};

use regex::{Regex,Captures};

pub struct Vars {
    pub t_max: f64,
    pub dim: Dim,
    pub phy: [f64;3],
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub dvars: Vec<(String,u32)>,
    pub noises: Option<Vec<Noises>>,
    pub parent: String,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback,Callback)>,
    vars: Vars,
}

impl Simulation {
    pub fn from_param<'a>(file_name: &'a str) -> gpgpu::Result<Self> {
        let param: Param = serde_yaml::from_str(&std::fs::read_to_string(file_name).expect(&format!("Could not find parameter file \"{}\".", file_name))).unwrap();
        let directory = std::path::Path::new(file_name);
        if let Some(directory) = directory.parent() {
            if directory.exists() {
                std::env::set_current_dir(&directory).expect(&format!("Could not change directory to \"{:?}\"",&directory));
            }
        }

        let mut handler = Handler::builder()?;

        if let Some(data_files) = &param.data_files {
            for f in data_files {
                let name = if let Some(i) = f.find('.') { &f[..i] } else { f };
                let name = if let Some(i) = name.rfind('/') { &name[i+1..] } else { name };
                handler = handler.load_data(name,Format::Column(&std::fs::read_to_string(f).expect(&format!("Could not find data file \"{}\".", f))),false,None); //TODO autodetect format from file extension
            }
        }

        let parent = if let Some(i) = file_name.find('.') { &file_name[..i] } else { file_name };
        let parent = if let Some(i) = parent.rfind('/') { &parent[i+1..] } else { parent };
        let target = std::path::Path::new(parent);
        std::fs::create_dir_all(&target).expect(&format!("Could not create destination directory \"{:?}\"", &target));
        extract_symbols(handler, param, parent.into())
    }

    pub fn run(&mut self) -> gpgpu::Result<()> {
        let Vars {t_max, dim, dirs: _, len, ref dvars, ref noises, phy: _ , parent: _} = self.vars;
        let noise_dim = |dim: &Option<usize>| D1(len*if let Some(d) = dim { *d } else { 1 }/2);//WARNING must divide by 2 because random number are computed 2 at a time
        let dvars = dvars.iter().map(|i| &i.0[..]).collect::<Vec<_>>();

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
            if let Some(noises) = noises {
                for noise in noises {
                    match noise {
                        Uniform{name,dim} => self.handler.run_arg("unifnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                        Normal{name,dim} => self.handler.run_arg("normnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                    }
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

fn extract_symbols(mut h: HandlerBuilder, mut param: Param, parent: String) -> gpgpu::Result<Simulation> {

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

    h = h.load_algorithm("moments");
    h = h.load_algorithm("sum");
    h = h.load_algorithm("correlation");
    h = h.load_algorithm("FFT");
    h = h.load_kernel_named("philox4x32_10_unit","unifnoise");
    h = h.load_kernel_named("philox4x32_10_normal","normnoise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");
    h = h.load_kernel("kc_times");
    h = h.load_kernel("kc_times_conj");
    h = h.load_kernel("ctimes");


    let mut consts = HashMap::new();
    for i in (0..3).filter(|&i| phy[i] != 0.0) {
        consts.insert(["dx","dy","dz"][i].to_string(),format!("{:e}",phy[i]/dims[i] as f64));
        consts.insert(["ivdx","ivdy","ivdz"][i].to_string(),format!("{:e}",dims[i] as f64/phy[i]));
    }

    if let Integrator::Euler{dt} = &param.integrator {
        consts.insert("dt".to_string(),format!("{:e}",dt));
        consts.insert("ivdt".to_string(),format!("{:e}",1.0/dt));
    }

    let (func,pdes,init) = parse_symbols(param.symbols, consts);
    for f in func {
        h = h.create_function(f);
    }

    let mut noises_names = HashSet::new();
    if let Some(noises) = &param.noises {
        for noise in noises {
            if !noises_names.insert(noise.name().to_string()) { panic!("Each noises must have a different name.") }
        }
    }
    let noises_names = noises_names.into_iter().collect::<Vec<_>>();

    if pdes.len() == 0 {
        panic!("PDEs must be given.")
    };
    let mut dvars = pdes.iter().map(|i| (i.dvar.clone(),i.expr.len())).collect::<Vec<_>>();
    match &param.integrator {
        Integrator::Euler{dt} => h = h.create_algorithm(create_euler_pde("integrator",*dt,pdes,Some(noises_names.clone()),vec![("t".into(),CF64)])),
        Integrator::QSS => panic!("QSS not handled yet."),
    }

    let mut init_kernels = vec![];
    if init.len() > 0 {
        let mut args = vec![KCBuffer("dst",CF64)];
        args.extend(dvars.iter().map(|pde| KCBuffer(&pde.0,CF64)));
        args.extend(noises_names.iter().map(|n| KCBuffer(&n,CF64)));
        let args: Vec<SKernelConstructor> = args.into_iter().map(|a| (&a).into()).collect();
        for ini in init {
            let len = dvars.iter().find(|(n,_)| n == &ini.name).expect(&format!("There must be a PDE corresponding to init data \"{}\"",&ini.name)).1;
            let name = format!("init_{}", &ini.name);
            h = h.create_kernel(gen_init_kernel(&name, len, args.clone(), ini));
            init_kernels.push(name);
        }
    }

    let mut init_file: HashMap<String,Vec<f64>> = if let Some(file) = param.initial_conditions_file {
        serde_yaml::from_str(
            &std::fs::read_to_string(&file)
            .expect(&format!("Could not find initial conditions file \"{}\".", &file))
        ).unwrap()
    } else {
        HashMap::new()
    };

    let mut max = 0;
    dvars.iter_mut().for_each(|i| {
        max = usize::max(max,i.1);
        i.0 = format!("dvar_{}",i.0); 
    });
    dvars.insert(0,("dvar_dst".to_string(),max));
    for dvar in &dvars {
        let data = init_file.remove(&dvar.0[5..])
            .and_then(|d| {
                if d.len() != len*dvar.1 {
                    panic!("Data stored in initial conditions file must have as much elements as the simulation total dim. Given \"{}\" of size {} whereas total dim is {}.", dvar.0, d.len(), len*dvar.1)
                }
                Some(Data(d.into()))
            }).unwrap_or(Len(F64(0.0),len*dvar.1));
        h = h.add_buffer(&dvar.0,data);
    }

    if init_file.len() > 0 { eprintln!("Warning, there are initial conditions that are not used from initial_conditions_file: {:?}.", init_file.keys()) }
    if let Some(noises) = &param.noises {
        for noise in noises {
            match noise {
                Uniform{name,dim} | Normal{name,dim} => {
                    let dim = dim.unwrap_or(1);
                    if dim == 0 { panic!("dim of noise \"{}\" must be different of 0.",name) }
                    let l = len*dim;
                    h = h.add_buffer(&format!("src{}",name),Len(U64(time),l));
                    time += 1;//TODO use U64_2 and another incrementer insted
                    h = h.add_buffer(name,Len(F64(0.0),l));
                    dvars.push((name.clone(),dim));
                },
            }
        }
    }
    let dvars = dvars.into_iter().map(|(n,d)| (n,d as u32)).collect::<Vec<_>>();

    if let Some(noises) = &mut param.noises {
        noises.iter_mut().for_each(|n| n.set_name(format!("src{}", n.name())));
    }

    h = h.add_buffer("tmp", Len(F64(0.0), len*max));
    h = h.add_buffer("sum", Len(F64(0.0), len*max));
    h = h.add_buffer("sumdst", Len(F64(0.0), lensum*max));
    h = h.add_buffer("moments", Len(F64(0.0), 4*max));
    h = h.add_buffer("srcFFT", Len(F64_2([0.0,0.0]), len*max));
    h = h.add_buffer("tmpFFT", Len(F64_2([0.0,0.0]), len*max));
    h = h.add_buffer("dstFFT", Len(F64_2([0.0,0.0]), len*max));
    h = h.add_buffer("initFFT", Len(F64_2([0.0,0.0]), len*max));

    let mut handler = h.build()?;
    if init_kernels.len()>0 {
        let noise_dim = |dim: &Option<usize>| D1(len*if let Some(d) = dim { *d } else { 1 }/2);//WARNING must divide by 2 because random number are computed 2 at a time
        if let Some(noises) = &param.noises {
            for noise in noises {
                match noise {
                    Uniform{name,dim} => handler.run_arg("unifnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                    Normal{name,dim} => handler.run_arg("normnoise", noise_dim(dim), &[BufArg(name,"src"),BufArg(&name[3..],"dst")])?,
                }
            }
        }
        let args = dvars.iter().map(|(n,_)| BufArg(n,if n.starts_with("dvar_") { &n[5..] } else { n })).collect::<Vec<_>>();
        for name in init_kernels {
            handler.run_arg(&name, dim, &args)?;
            handler.copy("dvar_dst", &name.replace("init_","dvar_"))?;
        }
    }

    let vars = Vars { t_max, dim, dirs, len, dvars, noises: param.noises, phy, parent };
    let callbacks = param.actions.into_iter().map(|(c,a)| (a.to_activation(),c.to_callback())).collect();

    Ok(Simulation { handler, callbacks, vars })
}

fn gen_func(name: String, args: Vec<(String,PrmType)>, src: String) -> SFunction {
    let args = args.into_iter().map(|a| match a.1 {
        Float => FCParam(a.0,CF64),
        Integer => FCParam(a.0,CU32),
        Indexable => FCGlobalPtr(a.0,CF64),
    }).collect::<Vec<_>>();

    SFunction {
        name,
        args,
        src: format!("return {};", src),
        ret_type: Some(CF64),
        needed: vec![],
    }
}

fn gen_init_kernel<'a>(name: &'a str, len: usize, args: Vec<SKernelConstructor>, ini: parameters::symbols::Init) -> SKernel {
    if len != ini.expr.len() {
        panic!("Then dim of the initial condition should be the same as the dpe, name: \"{}\"", name);
    }
    let mut id = "x+x_size*(y+y_size*z)".to_string();
    if len > 1 {
        id = format!("{}*({})", len, id);
    }
    let expr = if len == 1 {
        format!("    dst[_i] = {};\n", &ini.expr[0])
    } else {
        let mut expr = String::new();
        for i in 0..len {
            expr += &format!("    dst[{i}+_i] = {};\n", &ini.expr[i], i = i);
        }
        expr
    };
    SKernel {
        name: name.to_string(),
        args: args,
        src: format!("    uint _i = {};\n{}", id, expr),
        needed: vec![],
    }
}

fn parse_symbols(symbols: String, mut consts: HashMap<String,String>) -> (Vec<SFunction>,Vec<SPDE>,Vec<parameters::symbols::Init>) {
    let re = Regex::new(r"\b\w+\b").unwrap();
    let replace = |src: &str, consts: &HashMap<String,String>| re.replace_all(src, |caps: &Captures| {
        consts.get(&caps[0]).unwrap_or(&caps[0].to_string()).clone()
    }).to_string();

    let mut func = vec![];
    let mut pdes = vec![];
    let mut init = vec![];

    let search_const = Regex::new(r"^\s*(\w+)\s*(:?)=\s*(.+?)\s*$").unwrap();
    let search_func = Regex::new(r"^\s*(\w+)\((.+?)\)\s*(:?)=\s*(.+?)\s*$").unwrap();
    let search_pde = Regex::new(r"^\s*(\w+)'\s*(:?)=\s*(.+?)\s*$").unwrap();
    let search_init = Regex::new(r"^\s*\*(\w+)\s*(:?)=\s*(.+?)\s*$").unwrap();

    for l in symbols.lines() {
        if let Some(caps) = search_const.captures(l) {
            let name = caps[1].into();
            let src = if &caps[2] == ":" {
                caps[3].into()
            } else {
                panic!("Interpeter not supported yet.")
            };
            consts.insert(name,src);
        }
        if let Some(caps) = search_func.captures(l) {
            let name = caps[1].into();
            let args = caps[2].split(",").map(|i| {
                let val = i.trim().to_string();
                if val.starts_with("_") {
                    (val[1..].to_string(),Integer)
                } else if val.starts_with("*") {
                    (val[1..].to_string(),Indexable)
                } else {
                    (val,Float)
                }
            }).collect();
            let src = if &caps[3] == ":" {
                replace(&caps[4],&consts)
            } else {
                panic!("Interpeter not supported yet.")
            };
            func.push(gen_func(name,args,src))
        }
        if let Some(caps) = search_pde.captures(l) {
            let dvar = caps[1].into();
            let expr = if &caps[2] == ":" {
                let src = replace(&caps[3],&consts);
                if src.starts_with("(") && src.ends_with(")") {
                    src[1..src.len()-1].split(";").map(|i| i.trim().to_string()).collect()
                } else {
                    vec![src]
                }
            } else {
                panic!("Interpeter not supported yet.")
            };
            pdes.push(SPDE {dvar, expr});
        }
        if let Some(caps) = search_init.captures(l) {
            let name = caps[1].into();
            let expr = if &caps[2] == ":" {
                let src = replace(&caps[3],&consts);
                if src.starts_with("(") && src.ends_with(")") {
                    src[1..src.len()-1].split(";").map(|i| i.trim().to_string()).collect()
                } else {
                    vec![src]
                }
            } else {
                panic!("Interpeter not supported yet.")
            };
            init.push(parameters::symbols::Init {name, expr});
        }
    }

    pdes.iter_mut().for_each(|i| i.expr.iter_mut().for_each(|e| *e=replace(e,&consts)));

    (func,pdes,init)
}
