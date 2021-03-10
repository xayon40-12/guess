use gpgpu::algorithms::{AlgorithmParam::*, RandomType};
use gpgpu::data_file::Format;
use gpgpu::descriptors::{
    BufferConstructor::*, ConstructorTypes::*, KernelArg::*, KernelConstructor::*,
    SFunctionConstructor::*, SKernelConstructor, Types::*,
};
use gpgpu::functions::SFunction;
use gpgpu::integrators::pde_parser::DPDE;
use gpgpu::integrators::{
    create_euler_pde, create_projector_corrector_pde, create_rk4_pde, IntegratorParam, SPDE,
};
use gpgpu::kernels::SKernel;
use gpgpu::{handler::HandlerBuilder, Handler};
use gpgpu::{
    Dim::{self, *},
    DimDir,
};
use std::io::Write;

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod parameters;
use parameters::Noises::{self, *};
pub use parameters::{
    ActivationCallback, Callback, Init, Integrator, Param,
    PrmType::{self, *},
};

use regex::{Captures, Regex};

pub enum NumType {
    Single(usize),
    Multiple(usize, usize), // (quantity,start)
    NoNum,
}
use NumType::*;

pub struct Vars {
    pub t_max: f64,
    pub dim: Dim,
    pub phy: [f64; 3],
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub dvars: Vec<(String, u32)>,
    pub integrators: Vec<(String, Vec<String>)>,
    pub noises: Option<Vec<Noises>>,
    pub parent: String,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback, Callback)>,
    vars: Vars,
}

impl Simulation {
    pub fn from_param<'a>(file_name: &'a str, num: NumType, check: bool) -> gpgpu::Result<()> {
        let paramstr = std::fs::read_to_string(file_name)
            .expect(&format!("Could not find parameter file \"{}\".", file_name));
        let param: Param = match file_name
            .rfind('.')
            .and_then(|p| Some(&file_name[p + 1..]))
            .unwrap_or("")
        {
            "yaml" => serde_yaml::from_str(&paramstr).expect(&format!(
                "Could not convert yaml in file \"{}\".",
                file_name
            )),
            "ron" => ron::de::from_str(&paramstr)
                .expect(&format!("Could not convert ron in file \"{}\".", file_name)),
            a @ _ => panic!(
                "Unrecognised extension \"{}\" for parameter file \"{}\".",
                a, file_name
            ),
        };
        //let directory = std::path::Path::new(file_name);
        //if let Some(directory) = directory.parent() {
        //    if directory.exists() {
        //        std::env::set_current_dir(&directory).expect(&format!("Could not change directory to \"{:?}\"",&directory));
        //    }
        //}

        let mut handler = Handler::builder()?;
        //WARNING if there is no file extension but there is a "." in the path name, the behaviour
        //is wrong
        let parent = if let Some(i) = file_name.rfind('.') {
            &file_name[..i]
        } else {
            file_name
        }
        .to_string();
        if check {
            extract_symbols(handler, param, parent, true, 0)?;
            return Ok(());
        }

        let upparent = if let Some(i) = parent.rfind('/') {
            &parent[..i + 1]
        } else {
            ""
        }
        .to_string();
        if let Some(data_files) = &param.data_files {
            for f in data_files {
                let name = if let Some(i) = f.rfind('/') {
                    &f[i + 1..]
                } else {
                    f
                };
                let name = if let Some(i) = name.find('.') {
                    &name[..i]
                } else {
                    name
                };
                handler = handler.load_data(
                    name,
                    Format::Column(
                        &std::fs::read_to_string(&format!("{}{}", &upparent, f))
                            .expect(&format!("Could not find data file \"{}\".", f)),
                    ),
                    false,
                    None,
                ); //TODO autodetect format from file extension
            }
        }

        let run = |parent: &String, id: u64| -> gpgpu::Result<()> {
            let targetstr = format!("{}/config", parent);
            let target = std::path::Path::new(&targetstr);
            std::fs::create_dir_all(&target).expect(&format!(
                "Could not create destination directory \"{:?}\"",
                &target
            ));
            let dst = format!("{}/config/param.ron", parent);
            std::fs::write(&dst, &paramstr)
                .expect(&format!("Could not write parameter file to \"{}\"", dst));
            if let Some(mut sim) =
                extract_symbols(handler.clone(), param.clone(), parent.clone(), false, id)?
            {
                sim.run()?;
            }
            Ok(())
        };
        macro_rules! done {
            ($parent:ident) => {
                let mut done = std::fs::File::create(format!("{}/config/done", $parent))?;
                done.write_all(b"done")?;
            };
        }
        match num {
            Single(n) => {
                let parent = format!("{}{}", parent, n);
                run(&parent, n as _)?;
                done! {parent}
            }
            Multiple(n, start) => {
                for i in start..n + start {
                    let parent = format!("{}{}", parent, i);
                    run(&parent, i as _)?;
                    done! {parent}
                }
            }
            NoNum => {
                run(&parent, 0)?;
                done! {parent}
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> gpgpu::Result<()> {
        let Vars {
            t_max,
            dim,
            dirs: _,
            len,
            dvars: _,
            ref integrators,
            ref noises,
            phy: _,
            parent: _,
        } = self.vars;
        let noise_dim = |dim: &Option<usize>| D1(len * if let Some(d) = dim { *d } else { 1 });

        let mut intprm = IntegratorParam {
            t: 0.0,
            increment_name: "t".to_string(),
            args: vec![],
        };
        for (activator, callback) in &mut self.callbacks {
            if activator(intprm.t) {
                callback(&mut self.handler, &self.vars, intprm.t)?;
            }
        }
        #[cfg(debug_assertions)]
        let (mut pred, int) = (intprm.t, t_max / 100.0);
        while intprm.t < t_max {
            #[cfg(debug_assertions)]
            if intprm.t >= pred + int {
                pred = intprm.t;
                print!(" {}%\r", f64::trunc(intprm.t / int));
                std::io::stdout().lock().flush().unwrap();
            }
            if let Some(noises) = noises {
                for noise in noises {
                    match noise {
                        Uniform { name, dim, .. } => self.handler.run_algorithm(
                            "noise",
                            noise_dim(dim),
                            &[],
                            &[name, &name[3..]],
                            Ref(&RandomType::Uniform),
                        )?,
                        Normal { name, dim, .. } => self.handler.run_algorithm(
                            "noise",
                            noise_dim(dim),
                            &[],
                            &[name, &name[3..]],
                            Ref(&RandomType::Normal),
                        )?,
                    };
                }
            }

            //TODO run all integrator_i for each usb-process with the conresponding dvars
            for (integrator, vars) in integrators {
                let vars = vars.iter().map(|i| &i[..]).collect::<Vec<_>>();
                self.handler
                    .run_algorithm(integrator, dim, &[], &vars, Mut(&mut intprm))?;
            }

            for (activator, callback) in &mut self.callbacks {
                if activator(intprm.t) {
                    callback(&mut self.handler, &self.vars, intprm.t)?;
                }
            }
        }

        Ok(())
    }
}

fn extract_symbols(
    mut h: HandlerBuilder,
    mut param: Param,
    parent: String,
    check: bool,
    sim_id: u64,
) -> gpgpu::Result<Option<Simulation>> {
    let upparent = if parent.rfind('/').is_some() {
        format!(
            "{}/",
            if let Some(i) = parent.rfind('/') {
                &parent[..i]
            } else {
                &parent
            }
        )
    } else {
        "".to_string()
    };

    let (dims, phy) = param.config.dim.into();
    let dim: Dim = dims.into();
    let dirs = param.config.dirs.clone();
    let t_max = param.config.t_max;

    let mut sumdims = dims.clone();
    dirs.iter().for_each(|d| sumdims[*d as usize] = 1);
    let len = dims[0] * dims[1] * dims[2];
    let lensum = 2 * len / (sumdims[0] * sumdims[1] * sumdims[2]);
    let num = 4;
    let momsum = num * sumdims[0] * sumdims[1] * sumdims[2];

    //TODO verify that there is no link between two noise that start whith an initial condition
    //that differ of 1.
    let mut time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    h = h.load_algorithm("moments");
    h = h.load_algorithm("sum");
    h = h.load_algorithm("correlation");
    h = h.load_algorithm("FFT");
    h = h.load_algorithm_named("philox4x32_10", "noise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");
    h = h.load_kernel("kc_times");
    h = h.load_kernel("kc_times_conj");
    h = h.load_kernel("ctimes");
    h = h.load_kernel("cminus");
    h = h.load_kernel("vcminus");
    h = h.load_kernel("moments_to_cumulants");

    let mut consts = HashMap::new();
    for i in (0..3).filter(|&i| phy[i] != 0.0) {
        consts.insert(
            ["dx", "dy", "dz"][i].to_string(),
            format!("{:e}", phy[i] / dims[i] as f64),
        );
        consts.insert(
            ["ivdx", "ivdy", "ivdz"][i].to_string(),
            format!("{:e}", dims[i] as f64 / phy[i]),
        );
    }

    let nb_stages;
    {
        let dt = match &param.integrator {
            Integrator::Euler { dt } => {
                nb_stages = 1;
                dt
            }
            Integrator::PC { dt } => {
                nb_stages = 2;
                dt
            }
            Integrator::RK4 { dt } => {
                nb_stages = 4;
                dt
            }
        };

        consts.insert("dt".to_string(), format!("{:e}", dt));
        consts.insert("ivdt".to_string(), format!("{:e}", 1.0 / dt));
    }

    let mut noises_names = HashSet::new();
    let mut dpdes = param
        .fields
        .iter()
        .map(|f| DPDE {
            var_name: f.name.clone(),
            boundary: f.boundary.clone(),
            var_dim: dirs.len(),
            vec_dim: f.vect_dim.unwrap_or(1),
        })
        .collect::<Vec<_>>();
    if let Some(noises) = &param.noises {
        for noise in noises {
            if !noises_names.insert(noise.name().to_string()) {
                panic!("Each noises must have a different name.")
            }
            dpdes.push(DPDE {
                var_name: noise.name(),
                boundary: noise.boundary(),
                var_dim: dirs.len(),
                vec_dim: noise.dim().unwrap_or(1),
            })
        }
    }
    let noises_names = noises_names.into_iter().collect::<Vec<_>>();
    let (func, pdess, init) = parse_symbols(param.symbols, consts, dpdes);
    for f in func {
        h = h.create_function(f);
    }

    let mut dvars_names: HashSet<String> = HashSet::new();
    let mut dvars = vec![];
    let mut integrators = vec![];

    for pdes in &pdess {
        for i in pdes {
            if dvars_names.insert(i.dvar.clone()) {
                dvars.push((i.dvar.clone(), i.expr.len()));
            }
        }
    }
    for (i, pdes) in pdess.into_iter().enumerate() {
        if pdes.len() > 0 {
            let integrator = &format!("integrator_{}", i);
            let mut others = dvars
                .iter()
                .filter(|i| pdes.iter().filter(|j| j.dvar == i.0).count() == 0)
                .map(|i| i.0.clone())
                .collect::<Vec<_>>();
            let mut int_other_pdes = others
                .iter()
                .map(|i| format!("dvar_{}", i))
                .collect::<Vec<_>>();
            int_other_pdes.append(&mut noises_names.clone());
            others.append(&mut noises_names.clone());
            let pde_buffers = pdes.iter().map(|i| i.dvar.clone()).collect::<Vec<_>>();
            let nb_stages = match &param.integrator {
                Integrator::Euler { dt } => {
                    h = h.create_algorithm(create_euler_pde(
                        integrator,
                        *dt,
                        pdes,
                        Some(others),
                        vec![("t".into(), CF64)],
                    ));
                    1
                }
                Integrator::PC { dt } => {
                    h = h.create_algorithm(create_projector_corrector_pde(
                        integrator,
                        *dt,
                        pdes,
                        Some(others),
                        vec![("t".into(), CF64)],
                    ));
                    2
                }
                Integrator::RK4 { dt } => {
                    h = h.create_algorithm(create_rk4_pde(
                        integrator,
                        *dt,
                        pdes,
                        Some(others),
                        vec![("t".into(), CF64)],
                    ));
                    4
                }
            };
            let mut buffers = pde_buffers
                .iter()
                .flat_map(|name| {
                    let mut pde = vec![format!("dvar_{}", name)];
                    for j in 0..nb_stages {
                        pde.push(format!("tmp_dvar_{}_k{}", name, (j + 1)));
                    }
                    if nb_stages > 1 {
                        pde.push(format!("tmp_dvar_{}_tmp", name));
                    }
                    pde
                })
                .collect::<Vec<_>>();
            buffers.append(&mut int_other_pdes);
            integrators.push((integrator.to_string(), buffers));
        }
    }

    let mut init_kernels = vec![];
    if init.len() > 0 {
        let mut args = vec![KCBuffer("dst", CF64)];
        args.extend(dvars.iter().map(|pde| KCBuffer(&pde.0, CF64)));
        args.extend(noises_names.iter().map(|n| KCBuffer(&n, CF64)));
        let args: Vec<SKernelConstructor> = args.into_iter().map(|a| (&a).into()).collect();
        for ini in init {
            let len = dvars
                .iter()
                .find(|(n, _)| n == &ini.name)
                .expect(&format!(
                    "There must be a PDE corresponding to init data \"{}\"",
                    &ini.name
                ))
                .1;
            let name = format!("init_{}", &ini.name);
            h = h.create_kernel(gen_init_kernel(&name, len, args.clone(), ini));
            init_kernels.push(name);
        }
    }

    let mut init_file: HashMap<String, Vec<f64>> = if let Some(file) = param.initial_conditions_file
    {
        serde_yaml::from_str(
            &std::fs::read_to_string(&format!("{}{}", upparent, file)).expect(&format!(
                "Could not find initial conditions file \"{}\".",
                &file
            )),
        )
        .unwrap()
    } else {
        HashMap::new()
    };

    let mut max = 0;
    let mut name_to_index = HashMap::new();
    let mut index = 0;
    let num_pdes = dvars.len();
    dvars.iter_mut().for_each(|i| {
        max = usize::max(max, i.1);
        name_to_index.insert(i.0.clone(), index);
        index += if nb_stages > 1 { 2 } else { 1 } + nb_stages;
        i.0 = format!("dvar_{}", i.0);
    });
    for dvar in &dvars {
        let data = init_file.remove(&dvar.0[5..])
            .and_then(|d| {
                if d.len() != len*dvar.1 {
                    panic!("Data stored in initial conditions file must have as much elements as the simulation total dim. Given \"{}\" of size {} whereas total dim is {}.", dvar.0, d.len(), len*dvar.1)
                }
                Some(Data(d.into()))
            }).unwrap_or(Len(F64(0.0),len*dvar.1));
        if !check {
            h = h.add_buffer(&dvar.0, data);
            for i in 0..nb_stages {
                h = h.add_buffer(
                    &format!("tmp_{}_k{}", &dvar.0, (i + 1)),
                    Len(F64(0.0), len * dvar.1),
                );
            }
            if nb_stages > 1 {
                h = h.add_buffer(&format!("tmp_{}_tmp", &dvar.0), Len(F64(0.0), len * dvar.1));
            }
        }
    }
    if init_file.len() > 0 {
        eprintln!("Warning, there are initial conditions that are not used from initial_conditions_file: {:?}.", init_file.keys())
    }
    let renaming = |i: (String, usize)| {
        let mut vars = vec![i.clone()];
        for j in 0..nb_stages {
            vars.push((format!("tmp_{}_k{}", &i.0, (j + 1)), i.1));
        }
        if nb_stages > 1 {
            vars.push((format!("tmp_{}_tmp", &i.0), i.1));
        }
        vars.into_iter()
    };
    let mut dvars = dvars.into_iter().flat_map(renaming).collect::<Vec<_>>();

    if let Some(noises) = &param.noises {
        for noise in noises {
            match noise {
                Uniform { name, dim, .. } | Normal { name, dim, .. } => {
                    let dim = dim.unwrap_or(1);
                    if dim == 0 {
                        panic!("dim of noise \"{}\" must be different of 0.", name)
                    }
                    let l = len * dim;
                    if !check {
                        h = h
                            .add_buffer(&format!("src{}", name), Len(U64_2([time, sim_id]), l / 2));
                        h = h.add_buffer(name, Len(F64(0.0), l));
                    }
                    time += 1; //TODO use U64_2 and another incrementer insted
                    dvars.push((name.clone(), dim));
                    name_to_index.insert(name.clone(), index);
                    index += 1;
                    max = usize::max(max, dim);
                }
            }
        }
    }
    if check {
        println!("{}", h.source_code());
        return Ok(None);
    };
    let dvars = dvars
        .into_iter()
        .map(|(n, d)| (n, d as u32))
        .collect::<Vec<_>>();

    if let Some(noises) = &mut param.noises {
        noises
            .iter_mut()
            .for_each(|n| n.set_name(format!("src{}", n.name())));
    }

    h = h.add_buffer("tmp", Len(F64(0.0), len * max));
    h = h.add_buffer("tmp2", Len(F64(0.0), len * max));
    h = h.add_buffer("sum", Len(F64(0.0), len * max));
    h = h.add_buffer("sumdst", Len(F64(0.0), lensum * max));
    h = h.add_buffer("moments", Len(F64(0.0), momsum * max));
    h = h.add_buffer("cumulants", Len(F64(0.0), momsum * max));
    h = h.add_buffer("summoments", Len(F64(0.0), momsum * max));
    h = h.add_buffer("srcFFT", Len(F64_2([0.0, 0.0]), len * max));
    h = h.add_buffer("tmpFFT", Len(F64_2([0.0, 0.0]), len * max));
    h = h.add_buffer("dstFFT", Len(F64_2([0.0, 0.0]), len * max));
    h = h.add_buffer("initFFT", Len(F64_2([0.0, 0.0]), len * max));

    h = h.create_kernel(SKernel {
        name: "to_var".into(),
        args: vec![(&KCBuffer("src", CF64)).into()],
        src: "    src[x+x_size] = sqrt(src[x+x_size]-src[x]*src[x]);".into(),
        needed: vec![],
    });

    let mut handler = h.build()?;
    if init_kernels.len() > 0 {
        let noise_dim = |dim: &Option<usize>| D1(len * if let Some(d) = dim { *d } else { 1 });
        if let Some(noises) = &param.noises {
            for noise in noises {
                match noise {
                    Uniform { name, dim, .. } => handler.run_algorithm(
                        "noise",
                        noise_dim(dim),
                        &[],
                        &[name, &name[3..]],
                        Ref(&RandomType::Uniform),
                    )?,
                    Normal { name, dim, .. } => handler.run_algorithm(
                        "noise",
                        noise_dim(dim),
                        &[],
                        &[name, &name[3..]],
                        Ref(&RandomType::Normal),
                    )?,
                };
            }
        }
        let mut args = dvars
            .iter()
            .filter(|(n, _)| !n.starts_with("tmp_"))
            .map(|(n, _)| BufArg(n, if n.starts_with("dvar_") { &n[5..] } else { n }))
            .collect::<Vec<_>>();
        args.insert(0, BufArg("", ""));
        let init_kernels = init_kernels
            .iter()
            .map(|n| (n.clone(), format!("{}_k1", n.replace("init_", "tmp_dvar_"))))
            .collect::<Vec<_>>();
        for (name, swap_name) in &init_kernels {
            args[0] = BufArg(swap_name, "dst");
            handler.run_arg(name, dim, &args)?;
            handler.copy(&swap_name, &name.replace("init_", "dvar_"))?;
        }
    }

    let vars = Vars {
        t_max,
        dim,
        dirs,
        len,
        dvars,
        integrators,
        noises: param.noises,
        phy,
        parent,
    };
    let callbacks = param
        .actions
        .into_iter()
        .map(|(c, a)| (a.to_activation(), c.to_callback(&name_to_index, num_pdes)))
        .collect();

    Ok(Some(Simulation {
        handler,
        callbacks,
        vars,
    }))
}

fn gen_func(name: String, args: Vec<(String, PrmType)>, src: String) -> SFunction {
    let mut to_add: HashMap<String, String> = [
        ("x", "get_global_id(0)"),
        ("y", "get_global_id(1)"),
        ("z", "get_global_id(2)"),
        ("x_size", "get_global_size(0)"),
        ("y_size", "get_global_size(1)"),
        ("z_size", "get_global_size(2)"),
    ]
    .iter()
    .map(|(a, b)| (a.to_string(), b.to_string()))
    .collect();

    let args = args
        .into_iter()
        .map(|a| {
            to_add.remove(&a.0);
            match a.1 {
                Float => FCParam(a.0, CF64),
                Integer => FCParam(a.0, CU32),
                Indexable => FCGlobalPtr(a.0, CF64),
            }
        })
        .collect::<Vec<_>>();

    let mut globals = String::new();
    let mut to_add = to_add.iter().collect::<Vec<_>>();
    to_add.sort();
    for g in to_add {
        globals += &format!("    uint {} = {};\n", g.0, g.1);
    }

    SFunction {
        name,
        args,
        src: format!("{}    return {};", globals, src),
        ret_type: Some(CF64),
        needed: vec![],
    }
}

fn gen_init_kernel<'a>(
    name: &'a str,
    len: usize,
    args: Vec<SKernelConstructor>,
    ini: Init,
) -> SKernel {
    if len != ini.expr.len() {
        panic!(
            "Then dim of the initial condition should be the same as the dpe, name: \"{}\"",
            name
        );
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

fn parse_symbols(
    symbols: Vec<String>,
    mut consts: HashMap<String, String>,
    dpdes: Vec<DPDE>,
) -> (Vec<SFunction>, Vec<Vec<SPDE>>, Vec<Init>) {
    let re = Regex::new(r"\b\w+\b").unwrap();
    let replace = |src: &str, consts: &HashMap<String, String>| {
        re.replace_all(src, |caps: &Captures| {
            consts.get(&caps[0]).unwrap_or(&caps[0].to_string()).clone()
        })
        .to_string()
    };

    let mut func = vec![];
    let mut pdess = vec![];
    let mut init = vec![];

    let hdpdes = dpdes
        .iter()
        .map(|d| (&d.var_name, d))
        .collect::<HashMap<&String, &DPDE>>();

    let search_const = Regex::new(r"^\s*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_func = Regex::new(r"^\s*(\w+)\((.+?)\)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_pde = Regex::new(r"^\s*(\w+)'\s+(:?)=\s*(.+?)\s*$").unwrap();
    let _search_e = Regex::new(r"^\s*(\w+)|\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_init = Regex::new(r"^\s*\*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();

    use gpgpu::integrators::pde_parser::*;
    let mut lexer_idx = 0;

    for (i, symbols) in symbols.into_iter().enumerate() {
        let mut pdes = vec![];
        for (j, l) in symbols.lines().enumerate() {
            macro_rules! parse {
                ($src:ident) => {{
                    let mut parsed = parse(&dpdes, lexer_idx, &$src).expect(&format!(
                        "Parse error in sub-process {} line {}:\n|------------\n{}\n|------------\n",
                        i + 1,
                        j + 1,
                        &$src,
                    ));
                    lexer_idx += parsed.funs.len();
                    func.append(&mut parsed.funs);
                    parsed.ocl
                }};
            }
            let mut found = false;
            if let Some(caps) = search_const.captures(l) {
                let name = caps[1].into();
                let mut src = replace(&caps[3], &consts);
                if &caps[2] != ":" {
                    let mut res = parse!(src);
                    if res.len() == 1 {
                        src = res.pop().unwrap();
                    } else {
                        src = format!("[{}]", res.join(";"));
                    }
                }
                consts.insert(name, src);
                found = true;
            }
            if let Some(caps) = search_func.captures(l) {
                let name = caps[1].into();
                let args = caps[2]
                    .split(",")
                    .map(|i| {
                        let val = i.trim().to_string();
                        if val.starts_with("_") {
                            (val[1..].to_string(), Integer)
                        } else if val.starts_with("*") {
                            (val[1..].to_string(), Indexable)
                        } else {
                            (val, Float)
                        }
                    })
                    .collect();
                let mut src = replace(&caps[4], &consts);
                if &caps[3] != ":" {
                    let mut res = parse!(src);
                    if res.len() == 1 {
                        src = res.pop().unwrap();
                    } else {
                        panic!(
                        "The result of parsing a 'function' must be a single value and not a Vect."
                    );
                    }
                }
                func.push(gen_func(name, args, src));
                found = true;
            }
            if let Some(caps) = search_pde.captures(l) {
                let dvar: String = caps[1].into();
                let src = replace(&caps[3], &consts);
                let vec_dim = hdpdes.get(&dvar).expect(&format!("Unknown field \"{}\", it should be listed in the field \"fields\" in the parameter file.", &dvar)).vec_dim;
                let expr = if &caps[2] == ":" {
                    if src.starts_with("(") && src.ends_with(")") {
                        let expr = src[1..src.len() - 1]
                            .split(";")
                            .map(|i| i.trim().to_string())
                            .collect::<Vec<_>>();
                        if expr.len() != vec_dim {
                            panic!(
                                "The vectarial dim={} of '{}' is different from de dim={} parsed.",
                                vec_dim,
                                dvar,
                                expr.len()
                            );
                        }
                        expr
                    } else {
                        vec![src]
                    }
                } else {
                    parse!(src)
                };
                pdes.push(SPDE { dvar, expr });
                found = true;
            }
            if let Some(caps) = search_init.captures(l) {
                let name = caps[1].into();
                let src = replace(&caps[3], &consts);
                let expr = if &caps[2] == ":" {
                    if src.starts_with("(") && src.ends_with(")") {
                        src[1..src.len() - 1]
                            .split(";")
                            .map(|i| i.trim().to_string())
                            .collect()
                    } else {
                        vec![src]
                    }
                } else {
                    parse!(src)
                };
                init.push(Init { name, expr });
                found = true;
            }
            if !found {
                panic!(
                    "Line {} of sub-process {} could not be parsed.",
                    j + 1,
                    i + 1
                );
            }
        }

        pdes.iter_mut()
            .for_each(|i| i.expr.iter_mut().for_each(|e| *e = replace(e, &consts)));
        pdess.push(pdes);
    }

    (func, pdess, init)
}
