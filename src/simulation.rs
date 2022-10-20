use crate::gpgpu::algorithms::{AlgorithmParam::*, RandomType};
use crate::gpgpu::data_file::Format;
use crate::gpgpu::descriptors::{
    BufferConstructor::*, ConstructorTypes::*, KernelArg::*, KernelConstructor::*,
    SFunctionConstructor::*, SKernelConstructor, Types::*,
};
use crate::gpgpu::functions::SFunction;
use crate::gpgpu::integrators::{
    create_euler_pde, create_implicit_radau_pde, create_projector_corrector_pde, create_rk4_pde,
    CreatePDE, IntegratorParam, SPDE, STEP,
};
use crate::gpgpu::kernels::SKernel;
use crate::gpgpu::pde_parser::{pde_ir::Indexable, DPDE};
use crate::gpgpu::{handler::HandlerBuilder, Handler};
use crate::gpgpu::{
    Dim::{self, *},
    DimDir,
};
use std::io::Write;

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

pub mod parameters;
use parameters::Noises::{self, *};
pub use parameters::{
    ActivationCallback, Callback, EqDescriptor, Explicit, Implicit, Integrator, Param,
    PrmType::{self, *},
};

use regex::{Captures, Regex};

pub enum NumType {
    Single(usize),
    Multiple(usize, usize), // (quantity,start)
    NoNum,
}
use NumType::*;

#[derive(Debug)]
pub struct EquationKernel {
    kernel_name: String,
    args: Vec<(String, String)>,
    buf_var: String,
    buf_tmp: String,
}

pub struct Vars {
    pub t_0: f64,
    pub t_max: f64,
    pub dt_0: f64,
    pub dt_max: f64,
    pub dim: Dim,
    pub phy: [f64; 3],
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub dvars: Vec<(String, u32)>,
    pub stages: Vec<(Option<(String, Vec<String>)>, Vec<EquationKernel>)>,
    pub noises: Option<Vec<Noises>>,
    pub parent: String,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback, Callback)>,
    vars: Vars,
}

impl Simulation {
    pub fn from_param<'a>(
        file_name: &'a str,
        num: NumType,
        check: bool,
    ) -> crate::gpgpu::Result<()> {
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

        let run = |parent: &String, id: u64| -> crate::gpgpu::Result<()> {
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

    pub fn run(&mut self) -> crate::gpgpu::Result<()> {
        let Vars {
            t_0,
            t_max,
            dt_0,
            dt_max: _,
            dim,
            dirs: _,
            len,
            dvars: _,
            ref stages,
            ref noises,
            phy: _,
            parent: _,
        } = self.vars;
        let noise_dim = |dim: &Option<usize>| D1(len * if let Some(d) = dim { *d } else { 1 });

        let mut intprm = IntegratorParam {
            swap: 0,
            t: t_0,
            t_name: "t".to_string(),
            dt: dt_0,
            dt_name: "dt".to_string(),
            cdt_name: "cdt".to_string(),
            args: vec![],
        };
        for (activator, callback) in &mut self.callbacks {
            if activator(intprm.t) {
                callback(&mut self.handler, &self.vars, intprm.t)?;
            }
        }
        #[cfg(debug_assertions)]
        let (mut pred, int) = (intprm.t, t_max / 100.0);
        while intprm.t <= t_max {
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

            for (inte, equs) in stages {
                for equ in equs {
                    let mut args = equ
                        .args
                        .iter()
                        .map(|(b, n)| BufArg(b, n))
                        .collect::<Vec<_>>();
                    args.push(Param("t", intprm.t.into()));
                    self.handler.run_arg(&equ.kernel_name, dim, &args[..])?;
                }
                if let Some((integrator, vars)) = inte {
                    let vars = vars.iter().map(|i| &i[..]).collect::<Vec<_>>();
                    self.handler
                        .run_algorithm(integrator, dim, &[], &vars, Ref(&intprm))?;
                }
                for equ in equs {
                    self.handler.copy(&equ.buf_tmp, &equ.buf_var)?;
                }
            }
            //WARNING,TODO time t incremented only after each subprocesses, consider if it needs to be incremented per subprocess
            intprm.t += intprm.dt;
            intprm.swap = 1 - intprm.swap;

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
) -> crate::gpgpu::Result<Option<Simulation>> {
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
    let global_dim = dim.len();
    //let dirs = param.config.dirs.clone(); // FIXME isn't the physical non zero direction that are the dirs to consider?
    let dirs = phy.iter().enumerate().fold(vec![], |mut acc, (i, p)| {
        if *p == 0.0 {
            acc
        } else {
            acc.push(i.into());
            acc
        }
    });
    let t_0 = param.config.t_0.unwrap_or(0.0);
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
    h = h.load_function("ifNaNInf");

    let mut consts = HashMap::new();
    let mut dxyz = 1.0;
    for i in (0..3).filter(|&i| phy[i] != 0.0) {
        let d = phy[i] / dims[i] as f64;
        dxyz *= d;
        consts.insert(["dx", "dy", "dz"][i].to_string(), format!("{:e}", d));
        consts.insert(
            ["ivdx", "ivdy", "ivdz"][i].to_string(),
            format!("{:e}", 1.0 / d),
        );
    }
    consts.insert("dxyz".to_string(), dxyz.to_string());
    consts.insert("ivdxyz".to_string(), (1.0 / dxyz).to_string());

    let (nb_stages, dt_0, dt_max, _er, creator): (usize, f64, f64, f64, CreatePDE) =
        match &param.integrator {
            Integrator::Explicit(e) => match e {
                Explicit::Euler { dt } => (1, *dt, *dt, 0.0, create_euler_pde),
                Explicit::PC { dt } => (2, *dt, *dt, 0.0, create_projector_corrector_pde),
                Explicit::RK4 { dt } => (4, *dt, *dt, 0.0, create_rk4_pde),
            },
            Integrator::Implicit(i) => match i {
                Implicit::RadauIIA2 { dt_0, dt_max, er } => {
                    (2, *dt_0, *dt_max, *er, create_implicit_radau_pde)
                }
            },
        };

    let mut noises_names = HashSet::new();
    let mut dpdes = param
        .fields
        .unwrap_or(vec![])
        .iter()
        .map(|f| DPDE {
            var_name: f.name.clone(),
            boundary: f.boundary.clone().unwrap_or("periodic".into()),
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
    let default_boundary = if let Some(def) = param.default_boundary {
        def
    } else {
        "periodic".into()
    };
    let (func, pdess, init, equationss, constraintss) = parse_symbols(
        param.symbols,
        consts,
        default_boundary,
        dpdes,
        dirs.len(),
        global_dim,
    );
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
    for equs in &equationss {
        for i in equs {
            if dvars_names.insert(i.name.clone()) {
                dvars.push((i.name.clone(), i.expr.len()));
            } else {
                dvars.iter().for_each(|(n,l)| if n == &i.name && l != &i.expr.len() {
                    panic!("The equation \"{}\" should have the same vectorial length as the pde \"{}\"", &i.name, n);
                })
            }
        }
    }
    let nb_pdess = pdess.len();
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
            h = h.create_algorithm(creator(
                integrator,
                param.integrator.clone(),
                pdes,
                Some(others),
                vec![
                    ("t".into(), CF64),
                    ("dt".into(), CF64),
                    ("cdt".into(), CF64),
                ],
            ));
            let mut buffers = pde_buffers
                .iter()
                .flat_map(|name| {
                    let mut pde = vec![format!("dvar_{}", name)];
                    for j in 0..nb_stages {
                        pde.push(format!("tmp_dvar_{}_k{}", name, (j + 1)));
                    }
                    for j in 0..nb_stages {
                        pde.push(format!("tmp_sawp_dvar_{}_k{}", name, (j + 1)));
                    }
                    pde.push(format!("tmp_dvar_{}_tmp", name));
                    pde.push(format!("tmp_dvar_{}_tmp2", name));
                    pde
                })
                .collect::<Vec<_>>();
            buffers.append(&mut int_other_pdes);
            integrators.push(Some((integrator.to_string(), buffers)));
        } else {
            integrators.push(None);
        }
    }

    let mut init_kernels = vec![];

    let mut init_equ_args = vec![KCBuffer("dst", CF64)];
    init_equ_args.extend(dvars.iter().map(|pde| KCBuffer(&pde.0, CF64)));
    init_equ_args.extend(noises_names.iter().map(|n| KCBuffer(&n, CF64)));
    init_equ_args.push(KCParam("t", CF64));
    let init_equ_args: Vec<SKernelConstructor> =
        init_equ_args.into_iter().map(|a| a.into()).collect();

    if init.len() > 0 {
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
            h = h.create_kernel(gen_init_kernel(&name, len, init_equ_args.clone(), ini));
            init_kernels.push(name);
        }
    }

    for constraints in constraintss {
        for constraint in constraints {
            dvars
                .iter()
                .find(|(n, _)| n == &constraint.name)
                .expect(&format!(
                    "There must be a PDE corresponding to constraint \"{}\"",
                    &constraint.name
                ));
            let name = format!("constraint_{}", &constraint.name);
            h = h.create_kernel(gen_single_stage_kernel(
                &name,
                init_equ_args.clone(),
                constraint,
            ));
        }
    }

    let mut equation_kernelss = vec![];
    if equationss.len() > 0 {
        for (i, equs) in equationss.into_iter().enumerate() {
            let mut equation_kernels = vec![];
            for equ in equs {
                let name = equ.name.clone();
                let mut others = dvars
                    .iter()
                    .filter(|i| i.0 != name)
                    .map(|i| (format!("dvar_{}", &i.0), i.0.clone()))
                    .collect::<Vec<_>>();
                others.append(
                    &mut noises_names
                        .iter()
                        .map(|n| (n.to_string(), n.to_string()))
                        .collect::<Vec<_>>(),
                );
                let buf_var = format!("dvar_{}", &name);
                let buf_tmp = format!("tmp_dvar_{}_k1", &name);
                let mut args = vec![
                    (buf_var.clone(), name.clone()),
                    (buf_tmp.clone(), "dst".to_string()),
                ];
                args.append(&mut others);
                let kernel_name = format!("equ_{}_{}", &name, nb_pdess + i);
                h = h.create_kernel(gen_single_stage_kernel(
                    &kernel_name,
                    init_equ_args.clone(),
                    equ,
                ));
                equation_kernels.push(EquationKernel {
                    kernel_name,
                    args,
                    buf_var,
                    buf_tmp,
                });
            }
            equation_kernelss.push(equation_kernels);
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
        index += 2 + nb_stages;
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
                h = h.add_buffer(
                    &format!("tmp_sawp_{}_k{}", &dvar.0, (i + 1)),
                    Len(F64(0.0), len * dvar.1),
                );
            }
            h = h.add_buffer(&format!("tmp_{}_tmp", &dvar.0), Len(F64(0.0), len * dvar.1));
            h = h.add_buffer(
                &format!("tmp_{}_tmp2", &dvar.0),
                Len(F64(0.0), len * dvar.1),
            );
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
        vars.push((format!("tmp_{}_tmp", &i.0), i.1));
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
        args.push(Param("t", t_0.into()));
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

    let stages = integrators
        .into_iter()
        .zip(equation_kernelss.into_iter())
        .collect::<Vec<_>>();

    let vars = Vars {
        t_0,
        t_max,
        dt_0,
        dt_max,
        dim,
        dirs,
        len,
        dvars,
        stages,
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

fn gen_func(
    name: String,
    args: Vec<(String, PrmType)>,
    src: String,
    priors: Vec<String>,
) -> SFunction {
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
                Integer => FCParam(a.0, CI32),
                Indexable => FCGlobalPtr(a.0, CF64),
            }
        })
        .collect::<Vec<_>>();

    let mut globals = String::new();
    let mut to_add = to_add.iter().collect::<Vec<_>>();
    to_add.sort();
    for g in to_add {
        globals += &format!("    int {} = {};\n", g.0, g.1);
    }
    for p in priors {
        globals += &format!("    {}\n", p);
    }

    SFunction {
        name,
        args,
        src: format!("{}    return {};", globals, src),
        ret_type: Some(CF64),
        needed: vec![],
    }
}

fn gen_single_stage_kernel(
    name: &str,
    args: Vec<SKernelConstructor>,
    eqd: EqDescriptor,
) -> SKernel {
    let len = eqd.expr.len();
    let mut id = "x+x_size*(y+y_size*z)".to_string();
    if len > 1 {
        id = format!("{}*({})", len, id);
    }
    let expr = if len == 1 {
        format!("    dst[_i] = {};\n", &eqd.expr[0])
    } else {
        let mut expr = String::new();
        for i in 0..len {
            expr += &format!("    dst[{i}+_i] = {};\n", &eqd.expr[i], i = i);
        }
        expr
    };
    let priors = eqd.priors.join("\n    ");
    SKernel {
        name: name.to_string(),
        args,
        src: format!("    {}\n    uint _i = {};\n{}", priors, id, expr),
        needed: vec![],
    }
}

fn gen_init_kernel(
    name: &str,
    len: usize,
    args: Vec<SKernelConstructor>,
    ini: EqDescriptor,
) -> SKernel {
    if len != ini.expr.len() {
        panic!(
            "Then dim of the initial condition should be the same as the dpe, name: \"{}\"",
            name
        );
    }
    gen_single_stage_kernel(name, args, ini)
}

fn parse_symbols(
    mut symbols: Vec<String>,
    mut consts: HashMap<String, String>,
    default_boundary: String,
    mut dpdes: Vec<DPDE>,
    dim: usize,
    global_dim: usize,
) -> (
    Vec<SFunction>,
    Vec<Vec<SPDE>>,
    Vec<EqDescriptor>,
    Vec<Vec<EqDescriptor>>,
    Vec<Vec<EqDescriptor>>,
) {
    symbols.iter_mut().for_each(|s| {
        *s = s
            .lines()
            .map(|l| {
                if let Some(i) = l.find("//") {
                    &l[0..i]
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
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
    let mut equationss = vec![];
    let mut constraintss = vec![];

    let search_const = Regex::new(r"^\s*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_func = Regex::new(r"^\s*(\w+)\((.+?)\)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_pde = Regex::new(r"^\s*(\w+)'\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_e = Regex::new(r"^\s*(\w+)\|\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_constraint = Regex::new(r"^\s*(\w+)'c\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_eqpde = Regex::new(r"^\s*(\w+)'\|\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_betweenpde = Regex::new(r"^\s*(\w+)'(\d*)>\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_init = Regex::new(r"^\s*\*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_empty = Regex::new(r"^\s*$").unwrap();

    use crate::gpgpu::pde_parser::*;
    let mut lexer_idx = 0;

    for symbols in &symbols {
        for l in symbols.lines() {
            // search for pdes or expressions
            if let Some(caps) = search_pde
                .captures(l)
                .or(search_e.captures(l))
                .or(search_eqpde.captures(l))
                .or(search_betweenpde.captures(l))
            {
                let name = &caps[1];
                if dpdes.iter().filter(|i| i.var_name == name).count() == 0 {
                    // if a pde is not referenced add it to the known pdes with default
                    // vec_dim of one and the default boundary function
                    dpdes.push(DPDE {
                        var_name: name.into(),
                        var_dim: dim,
                        vec_dim: 1,
                        boundary: default_boundary.clone(),
                    })
                }
            }
        }
    }

    let hdpdes = dpdes
        .iter()
        .map(|d| (&d.var_name, d))
        .collect::<HashMap<&String, &DPDE>>();

    let choice = |c: [&str; 3]| {
        c.iter()
            .take(global_dim)
            .map(|s| s.to_string())
            .collect::<String>()
    };
    symbols.insert(
        0,
        format!(
            "modx := ((x+x_size)%x_size)
    mody := ((y+y_size)%y_size)
    modz := ((z+z_size)%z_size)
    periodic1(_x,_w,_w_size,*u) := u[w + w_size*modx]
    periodic2(_x,_y,_w,_w_size,*u) := u[w + w_size*(modx + x_size*mody)]
    periodic3(_x,_y,_z,_w,_w_size,*u) := u[w + w_size*(modx + x_size*(mody + y_size*modz))]
    periodic({c}_w,_w_size,*u) := periodic{n}({p}w,w_size,u)
    ghostx := (x<0 ? -x : (x>=x_size ? 2*(x_size-1)-x : x))
    ghosty := (y<0 ? -y : (y>=y_size ? 2*(y_size-1)-y : y))
    ghostz := (z<0 ? -z : (z>=z_size ? 2*(z_size-1)-z : z))
    ghost1(_x,_w,_w_size,*u) := u[w + w_size*ghostx]
    ghost2(_x,_y,_w,_w_size,*u) := u[w + w_size*(ghostx + x_size*ghosty)]
    ghost3(_x,_y,_z,_w,_w_size,*u) := u[w + w_size*(ghostx + x_size*(ghosty + y_size*ghostz))]
    ghost({c}_w,_w_size,*u) := ghost{n}({p}w,w_size,u)",
            c = choice(["_x,", "_y,", "_z,"]),
            n = global_dim,
            p = choice(["x,", "y,", "z,"])
        ),
    );
    for (i, symbols) in symbols.into_iter().enumerate() {
        let mut pdes = vec![];
        let mut constraints = vec![];
        let mut equations = vec![];

        macro_rules! parse {
            ($j:ident $nj:ident, $src:ident, $current_var:expr, $compact:expr) => {{
                let mut parsed = parse(
                    &dpdes,
                    &$current_var.and_then(|name: &String| {
                        hdpdes.get(name).and_then(|pde| {
                            Some(if pde.vec_dim > 1 {
                                Indexable::new_vector(
                                    pde.var_dim,
                                    global_dim,
                                    pde.vec_dim,
                                    name,
                                    &pde.boundary,
                                )
                            } else {
                                Indexable::new_scalar(pde.var_dim, global_dim, name, &pde.boundary)
                            })
                        })
                    }),
                    lexer_idx,
                    global_dim,
                    &$src,
                    $compact,
                )
                .expect(&format!(
                    "Parse error in sub-process {} line{}:\n|------------\n{}\n|------------\n",
                    i,
                    if $j == $nj {
                        format!(" {}", $j + 1)
                    } else {
                        format!("s {}-{}", $j, $nj)
                    },
                    &$src,
                ));
                lexer_idx += parsed.funs.len();
                func.append(&mut parsed.funs);
                (parsed.ocl, parsed.priors)
            }};
        }

        let mut doit = |j, nj, l: &str| {
            let mut found = false;
            if let Some(caps) = search_const.captures(&l) {
                let name = caps[1].into();
                let mut src = replace(&caps[3], &consts);
                if &caps[2] != ":" {
                    let mut res = parse!(j nj, src, None, false).0;
                    if res.len() == 1 {
                        src = format!("({})", res.pop().unwrap());
                    } else {
                        src = format!("[{}]", res.join(";"));
                    }
                }
                consts.insert(name, src);
                found = true;
            }
            if let Some(caps) = search_func.captures(&l) {
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
                let mut priors = None;
                if &caps[3] != ":" {
                    let (mut res, p) = parse!(j nj, src, None, true);
                    priors = Some(p);
                    if res.len() == 1 {
                        src = res.pop().unwrap();
                    } else {
                        panic!(
                        "The result of parsing a 'function' must be a single value and not a Vect."
                    );
                    }
                }
                func.push(gen_func(name, args, src, priors.unwrap_or_default()));
                found = true;
            }
            macro_rules! beg {
                ($search:ident $arr:ident, $name:ident, $expr:ident, $priors:ident, $val:expr) => {
                    if let Some(caps) = $search.captures(&l) {
                        let $name = caps[1].into();
                        let src = replace(&caps[3], &consts);
                let vec_dim = hdpdes.get(&$name).expect(&format!("Unknown field \"{}\", it should be listed in the field \"fields\" in the parameter file.", &$name)).vec_dim;
                        let ($expr,$priors) = if &caps[2] == ":" {
                            if src.starts_with("(") && src.ends_with(")") {
                                let expr = src[1..src.len() - 1]
                                    .split(";")
                                    .map(|i| i.trim().to_string())
                                    .collect::<Vec<_>>();
                                if expr.len() != vec_dim {
                                    panic!(
                                        "The vectorial dim={} of '{}' is different from de dim={} parsed.",
                                        vec_dim,
                                        $name,
                                        expr.len()
                                    );
                                }
                                (expr,vec![])
                            } else {
                                (vec![src],vec![])
                            }
                        } else {
                            parse!(j nj, src, Some(&$name), true)
                        };
                        $arr.push($val);
                        found = true;
                    }
                };
                ($search:ident $arr:ident, $name:ident, $expr:ident, $priors:ident, $i:ident, $val:expr) => {
                    if let Some(caps) = $search.captures(&l) {
                        let $name = caps[1].into();
                        let $i = caps[2].parse::<u32>().unwrap();
                        let src = replace(&caps[4], &consts);
                let vec_dim = hdpdes.get(&$name).expect(&format!("Unknown field \"{}\", it should be listed in the field \"fields\" in the parameter file.", &$name)).vec_dim;
                        let ($expr,$priors) = if &caps[3] == ":" {
                            if src.starts_with("(") && src.ends_with(")") {
                                let expr = src[1..src.len() - 1]
                                    .split(";")
                                    .map(|i| i.trim().to_string())
                                    .collect::<Vec<_>>();
                                if expr.len() != vec_dim {
                                    panic!(
                                        "The vectorial dim={} of '{}' is different from de dim={} parsed.",
                                        vec_dim,
                                        $name,
                                        expr.len()
                                    );
                                }
                                (expr,vec![])
                            } else {
                                (vec![src],vec![])
                            }
                        } else {
                            parse!(j nj, src, Some(&$name),true)
                        };
                        $arr.push($val);
                        found = true;
                    }
                };
            }
            macro_rules! equ {
                ($search:ident $arr:ident) => {
                    beg!($search $arr, name, expr, priors, EqDescriptor { name, expr, priors });
                };
                ($search:ident $arr:ident, $step:expr) => {
                    beg!($search $arr, name, expr, priors, SPDE {dvar: name,expr, priors,step: $step, constraint: None});
                };
                ($search:ident $arr:ident, $i:ident $step:expr) => {
                    beg!($search $arr, name, expr, priors, $i, SPDE {dvar: name,expr, priors,step: $step, constraint: None});
                };
            }
            equ! {search_init init}
            equ! {search_pde pdes, STEP::PDE}
            equ! {search_eqpde pdes, STEP::EQPDE}
            equ! {search_betweenpde pdes, i STEP::BETWEENPDE(i)}
            equ! {search_e equations}
            equ! {search_constraint constraints}
            if !found && !search_empty.is_match(l) {
                panic!("Line {} of sub-process {} could not be parsed.", j + 1, i);
            }
        };

        let mut l = String::new();
        let mut first = false;
        let mut j = 0;
        let mut nj = 0;
        for (cj, cl) in symbols.lines().enumerate() {
            if cl.contains('=') {
                if !first {
                    doit(j, nj, &l);
                } else {
                    first = true;
                }
                l = cl.into();
                j = cj;
            } else {
                l += cl;
            }
            nj = cj;
        }
        doit(j, nj, &l);
        constraints.sort_by_key(|EqDescriptor { name, .. }| name.clone());
        pdes.iter_mut().for_each(|i| {
            i.expr.iter_mut().for_each(|e| *e = replace(e, &consts));
            if let Ok(constraint_id) =
                constraints.binary_search_by_key(&i.dvar, |EqDescriptor { name, .. }| name.clone())
            {
                i.constraint = Some(format!("constraint_{}", constraints[constraint_id].name));
            }
        });
        constraints
            .iter_mut()
            .for_each(|i| i.expr.iter_mut().for_each(|e| *e = replace(e, &consts)));
        equations
            .iter_mut()
            .for_each(|i| i.expr.iter_mut().for_each(|e| *e = replace(e, &consts)));
        pdess.push(pdes);
        constraintss.push(constraints);
        equationss.push(equations);
    }

    (func, pdess, init, equationss, constraintss)
}
