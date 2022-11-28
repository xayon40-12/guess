use crate::concurrent_hdf5::AttributeType;
use crate::concurrent_hdf5::ConcurrentHDF5;
use crate::gpgpu::algorithms::{AlgorithmParam::*, RandomType};
use crate::gpgpu::data_file::Format;
use crate::gpgpu::descriptors::{
    BufferConstructor::*, ConstructorTypes::*, KernelArg::*, KernelConstructor::*,
    SFunctionConstructor::*, SKernelConstructor, Types::*,
};
use crate::gpgpu::functions::SFunction;
use crate::gpgpu::integrators::{
    create_euler_pde, create_implicit_euler_pde, create_implicit_radau_pde,
    create_projector_corrector_pde, create_rk4_pde, CreatePDE, IntegratorParam, SPDE, STEP,
};
use crate::gpgpu::kernels::SKernel;
use crate::gpgpu::pde_parser::pde_ir::coords_str;
use crate::gpgpu::pde_parser::{pde_ir::Indexable, DPDE};
use crate::gpgpu::{handler::HandlerBuilder, Handler};
use crate::gpgpu::{
    Dim::{self, *},
    DimDir,
};
use std::any::Any;
use std::io::Write;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub mod parameters;
use hdf5::types::VarLenUnicode;
use itertools::Itertools;
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

use self::parameters::InitFormat;
use self::parameters::OutputType;

#[derive(Debug, Clone)]
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
    pub dt_factor: f64,
    pub dt_reset: f64,
    pub max_iter: usize,
    pub max_reset: usize,
    pub nb_propagate: usize,
    pub dim: Dim,
    pub phy: [f64; 3],
    pub dirs: Vec<DimDir>,
    pub len: usize,
    pub dvars: Vec<(String, u32)>,
    pub stage: (Option<(String, Vec<String>)>, Vec<EquationKernel>),
    pub noises: Option<Vec<Noises>>,
    pub parent: String,
}

pub struct Simulation {
    handler: Handler,
    callbacks: Vec<(ActivationCallback, Callback)>,
    vars: Vars,
}

macro_rules! update {
    ($storage:ident, $path:expr, $attr:expr, $v:expr, $i:expr) => {
        if let Err(e) = $storage.update_attr($path, $attr, AttributeType::Group, |i| i, $v) {
            eprintln!(
                "Error in update \"{}\" attribute in hdf5 file for simulation number {}:\n{:?}",
                $attr, $i, e
            );
        }
    };
}

impl Simulation {
    pub fn from_param<'a>(
        file_name: &'a str,
        num: NumType,
        total_to_fuse: Option<usize>,
        check: bool,
    ) -> crate::gpgpu::Result<()> {
        let paramstr = std::fs::read_to_string(file_name)
            .expect(&format!("Could not find parameter file \"{}\".", file_name));
        //let directory = std::path::Path::new(file_name);
        //if let Some(directory) = directory.parent() {
        //    if directory.exists() {
        //        std::env::set_current_dir(&directory).expect(&format!("Could not change directory to \"{:?}\"",&directory));
        //    }
        //}

        //WARNING if there is no file extension but there is a "." in the path name, the behaviour
        //is wrong
        let parent = if let Some(i) = file_name.rfind('.') {
            &file_name[..i]
        } else {
            file_name
        }
        .to_string();

        let upparent = if let Some(i) = parent.rfind('/') {
            &parent[..i + 1]
        } else {
            ""
        }
        .to_string();
        macro_rules! param {
            ($param:ident, $paramstr:expr) => {
                let $param: Param = match file_name
                    .rfind('.')
                    .and_then(|p| Some(&file_name[p + 1..]))
                    .unwrap_or("")
                {
                    "yaml" => serde_yaml::from_str(&$paramstr).expect(&format!(
                        "Could not convert yaml in file \"{}\".",
                        file_name
                    )),
                    "ron" => ron::de::from_str(&$paramstr)
                        .expect(&format!("Could not convert ron in file \"{}\".", file_name)),
                    a @ _ => panic!(
                        "Unrecognised extension \"{}\" for parameter file \"{}\".",
                        a, file_name
                    ),
                };
            };
        }

        if check {
            param!(param, paramstr);
            let handler = Handler::builder()?;
            extract_symbols(handler, param, paramstr, parent, true, 0)?;
            return Ok(());
        }
        let run = |parent: &String, id: u64| -> crate::gpgpu::Result<IntegratorParam> {
            let paramstr = paramstr.replace("@ID", &id.to_string());
            //let mut handler = handler.clone();
            let mut handler = Handler::builder()?;
            param!(param, paramstr);
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
            let parent = &format!("{}/{}", parent, id);
            let (mut sim, hdf5_file) =
                extract_symbols(handler, param.clone(), paramstr, parent.clone(), false, id)?
                    .expect("Unexpected error: no Simulation available after extracting symbols");

            sim.run(total_to_fuse, hdf5_file, id)
        };

        match num {
            Single(i) => {
                run(&parent, i as _)?; //TODO provide parent without the id
            }
            Multiple(n, start) => {
                for i in start..n + start {
                    run(&parent, i as _)?;
                }
            }
            NoNum => {
                let i = 0;
                run(&parent, i)?;
            }
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        total_to_fuse: Option<usize>,
        mut hdf5_file: Option<ConcurrentHDF5>,
        id: u64,
    ) -> crate::gpgpu::Result<IntegratorParam> {
        let t_start = Instant::now();
        let Vars {
            t_0,
            t_max,
            dt_0,
            dt_max,
            dt_factor,
            dt_reset,
            max_iter,
            max_reset,
            nb_propagate,
            dim,
            dirs: _,
            len,
            dvars: _,
            ref stage,
            ref noises,
            phy: _,
            ref parent,
        } = self.vars;
        let noise_dim = |dim: &Option<usize>| D1(len * if let Some(d) = dim { *d } else { 1 });

        let mut intprm = IntegratorParam {
            swap: 0,
            t: t_0,
            t_name: "t".to_string(),
            dt: dt_0,
            dt_max,
            dt_factor,
            dt_reset,
            max_iter,
            max_reset,
            count: 0.0,
            nb_propagate,
            dt_name: "dt".to_string(),
            cdt_name: "cdt".to_string(),
            args: vec![],
        };
        for (activator, callback) in &mut self.callbacks {
            if activator(intprm.t) <= 0.0 {
                callback(&mut self.handler, &self.vars, &mut hdf5_file, intprm.t)?;
            }
        }
        #[cfg(debug_assertions)]
        let (mut pred, int) = (intprm.t, t_max / 100.0);
        let mut dtsave = None;
        let mut tdtcount = vec![(intprm.t, intprm.dt, intprm.count)];
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

            let mut update: Option<Box<dyn Any>> = None;
            let (inte, equs) = &stage;
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
                update = self
                    .handler
                    .run_algorithm(integrator, dim, &[], &vars, Ref(&intprm))?;
            }
            for equ in equs {
                self.handler.copy(&equ.buf_tmp, &equ.buf_var)?;
            }
            intprm = *update
                .expect("Integrator algorithm must return the next IntegratorParam.")
                .downcast()
                .expect("Integrator algorithm did not return a proper IntegratorParam type.");
            tdtcount.push((intprm.t, intprm.dt, intprm.count));

            let mut md = f64::MAX;
            for (activator, callback) in &mut self.callbacks {
                let d = activator(intprm.t);
                if d >= 0.0 && d < intprm.t * 1e-15 {
                    callback(&mut self.handler, &self.vars, &mut hdf5_file, intprm.t)?;
                } else if d > 0.0 {
                    md = md.min(d);
                }
            }
            if md < intprm.dt {
                if dtsave.is_none() {
                    dtsave = Some(intprm.dt);
                }
                intprm.dt = md;
            } else if md < 2.0 * intprm.dt {
                if dtsave.is_none() {
                    dtsave = Some(intprm.dt);
                }
                intprm.dt = md / 2.0;
            } else if let Some(dt) = dtsave {
                intprm.dt = dt;
                dtsave = None;
            }
        }

        let t_end = t_start.elapsed().as_secs_f32();
        let count = intprm.count;
        if let Some(storage) = &mut hdf5_file {
            match storage.update_attr(&parent, "nb_done", AttributeType::Group, |i| i + 1, &0) {
                Err(e) => eprintln!(
                    "Error in update \"done_count\" attribute in hdf5 file:\n{:?}",
                    e
                ),
                Ok(current_tot) => {
                    if let Some(tot) = total_to_fuse {
                        if current_tot == tot {
                            // TODO: fuse
                        }
                    }
                }
            }
            update!(storage, &parent, "done", &true, id);
            update!(storage, &parent, "elapsed", &t_end, id);
            update!(storage, &parent, "count", &count, id);
        } else {
            let mut done = std::fs::File::create(format!("{}/config/done", parent))?;
            done.write_all(&format!("elapsed: {}\ncount: {}", t_end, count).as_bytes())?;
            let mut tdtcount_file = std::fs::File::create(format!("{}/config/tdtcount", parent))?;
            let tdtcount = tdtcount
                .into_iter()
                .map(|(t, dt, count)| format!("{} {} {}", t, dt, count))
                .collect::<Vec<_>>()
                .join("\n");
            tdtcount_file.write_all(&tdtcount.as_bytes())?;
        }
        Ok(intprm)
    }
}

fn extract_symbols(
    mut h: HandlerBuilder,
    mut param: Param,
    paramstr: String,
    parent: String,
    check: bool,
    sim_id: u64,
) -> crate::gpgpu::Result<Option<(Simulation, Option<ConcurrentHDF5>)>> {
    let (parent, mut hdf5_file) = match param.output {
        Some(OutputType::HDF5(hdf5name)) => (parent, ConcurrentHDF5::new(&hdf5name).ok()),
        Some(OutputType::Text(path)) => (path, None),
        None => (parent, None),
    };

    let parent_no_id = &parent[..parent.rfind('/').expect("This is a bug: There must be a '/' in the current folder name that separate the simulation name from its id.")];
    let upparent = parent_no_id
        .rfind('/')
        .and_then(|i| Some(&parent[..i + 1]))
        .unwrap_or("")
        .to_string();

    if !check {
        if let Some(hdf5) = &mut hdf5_file {
            let unicode =
                VarLenUnicode::from_str(&paramstr).expect("Inalide unicode in parameter file.");
            update!(hdf5, &parent, "param", &unicode, sim_id);
        } else {
            let targetstr = format!("{}/config", parent);
            let target = std::path::Path::new(&targetstr);
            std::fs::create_dir_all(&target).expect(&format!(
                "Could not create destination directory \"{:?}\"",
                &target
            ));
            let dst = format!("{}/config/param.ron", parent);
            std::fs::write(&dst, &paramstr)
                .expect(&format!("Could not write parameter file to \"{}\"", dst));
        }
    }

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
    let mlen = len.max(lensum).max(momsum);

    //TODO verify that there is no link between two noise that start whith an initial condition
    //that differ of 1.
    let mut time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    h = h.load_algorithm("moments");
    h = h.load_algorithm("sum");
    h = h.load_algorithm("max");
    h = h.load_algorithm("correlation");
    //h = h.load_algorithm("FFT");
    h = h.load_algorithm_named("philox4x32_10", "noise");
    h = h.load_kernel("complex_from_real");
    h = h.load_kernel("kc_sqrmod");
    h = h.load_kernel("kc_times");
    h = h.load_kernel("kc_times_conj");
    h = h.load_kernel("ctimes");
    h = h.load_kernel("cdivides");
    h = h.load_kernel("cminus");
    h = h.load_kernel("vcminus");
    h = h.load_kernel("moments_to_cumulants");
    h = h.load_kernel("anisotropy");
    h = h.load_function("ifNaNInf");
    h = h.load_function("ifelse");
    h = h.load_function("lessthan");
    h = h.load_function("fmaxNaNInf");

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

    let default_dt_reset = 0.5;
    let default_dt_factor = 1.1;
    let default_max_iter = 10;
    let default_max_reset = 100;
    let implicit = match &param.integrator {
        Integrator::Explicit { .. } => false,
        Integrator::Implicit { .. } => true,
    };
    let (nb_stages, dt_0, dt_max, dt_factor, dt_reset, max_iter, max_reset, _er, creator): (
        usize,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        f64,
        CreatePDE,
    ) = match &param.integrator {
        Integrator::Explicit { dt, er, scheme } => {
            let er = er.unwrap_or(0.0);
            match scheme {
                Explicit::Euler => (
                    1,
                    *dt,
                    *dt,
                    default_dt_factor,
                    default_dt_reset,
                    default_max_iter,
                    default_max_reset,
                    er,
                    create_euler_pde,
                ),
                Explicit::PC => (
                    2,
                    *dt,
                    *dt,
                    default_dt_factor,
                    default_dt_reset,
                    default_max_iter,
                    default_max_reset,
                    er,
                    create_projector_corrector_pde,
                ),
                Explicit::RK4 => (
                    4,
                    *dt,
                    *dt,
                    default_dt_factor,
                    default_dt_reset,
                    default_max_iter,
                    default_max_reset,
                    er,
                    create_rk4_pde,
                ),
            }
        }
        Integrator::Implicit {
            dt_0,
            dt_max,
            dt_factor,
            dt_reset,
            max_iter,
            max_reset,
            er,
            scheme,
        } => match scheme {
            Implicit::RadauIIA2 => (
                2,
                dt_0.unwrap_or(*dt_max),
                *dt_max,
                dt_factor.unwrap_or(default_dt_factor),
                dt_reset.unwrap_or(default_dt_reset),
                max_iter.unwrap_or(default_max_iter),
                max_reset.unwrap_or(default_max_reset),
                *er,
                create_implicit_radau_pde,
            ),

            Implicit::Euler => (
                1,
                dt_0.unwrap_or(*dt_max),
                *dt_max,
                dt_factor.unwrap_or(default_dt_factor),
                dt_reset.unwrap_or(default_dt_reset),
                max_iter.unwrap_or(default_max_iter),
                max_reset.unwrap_or(default_max_reset),
                *er,
                create_implicit_euler_pde,
            ),
        },
    };

    let default_boundary = "ghost";
    let mut noises_names = HashSet::new();
    let mut dpdes = param
        .fields
        .unwrap_or(vec![])
        .iter()
        .map(|f| DPDE {
            var_name: f.name.clone(),
            boundary: f.boundary.clone().unwrap_or(default_boundary.into()),
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
        default_boundary.into()
    };
    let boundaries = dpdes
        .iter()
        .map(|DPDE { boundary, .. }| boundary.to_string())
        .chain([default_boundary.clone()])
        .unique()
        .collect::<Vec<_>>();
    let (func, pdes, init, equations, constraints, max_space_derivative_depth) = parse_symbols(
        param.symbols,
        consts,
        default_boundary,
        dpdes,
        dirs.len(),
        global_dim,
    );
    let nb_propagate = (1 + max_space_derivative_depth) / 2 + 1; // +1 so that the error is farther handled
    for f in func {
        h = h.create_function(f);
    }

    let mut dvars_names: HashSet<String> = HashSet::new();
    let mut dvars = vec![];
    let integrator;

    for i in &pdes {
        if dvars_names.insert(i.dvar.clone()) {
            dvars.push((i.dvar.clone(), i.expr.len()));
        }
    }
    for i in &equations {
        if dvars_names.insert(i.name.clone()) {
            dvars.push((i.name.clone(), i.expr.len()));
        } else {
            dvars.iter().for_each(|(n,l)| if n == &i.name && l != &i.expr.len() {
                    panic!("The equation \"{}\" should have the same vectorial length as the pde \"{}\"", &i.name, n);
                })
        }
    }
    let pure_constraints = constraints
        .iter()
        .filter(|(_, pure)| *pure)
        .map(|(c, _)| (format!("constraint_{}", &c.name), c.expr.len()))
        .collect::<Vec<_>>();
    let mut eqpde_init_copy = vec![];
    if pdes.len() > 0 {
        let integrator_name = &format!("integrator_{}", 0);
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
        for pde in &pdes {
            let name = pde.dvar.clone();
            match pde.step {
                STEP::BETWEENPDE(_) | STEP::EQPDE => {
                    for j in 0..nb_stages {
                        eqpde_init_copy.push((
                            format!("dvar_{}", name.clone()),
                            format!("tmp_dvar_{}_k{}", name.clone(), (j + 1)),
                        ));
                    }
                }
                _ => {}
            }
        }
        h = h.create_algorithm(creator(
            integrator_name,
            param.integrator.clone(),
            &pdes,
            &pure_constraints,
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
                    pde.push(format!("tmp_swap_dvar_{}_k{}", name, (j + 1)));
                }
                for j in 0..nb_stages {
                    pde.push(format!("tmp_save_dvar_{}_k{}", name, (j + 1)));
                }
                pde.push(format!("tmp_dvar_{}_tmp", name));
                pde
            })
            .chain(["tmp_error".to_string()])
            .chain(pure_constraints.iter().map(|(name, _)| name.clone()))
            .collect::<Vec<_>>();
        buffers.append(&mut int_other_pdes);
        integrator = Some((integrator_name.to_string(), buffers));
    } else {
        integrator = None;
    }

    let mut init_kernels = vec![];

    let vars_names = dvars
        .iter()
        .map(|pde| pde.0.clone())
        .collect::<Vec<String>>();
    let o_vars_names = vars_names
        .iter()
        .map(|v| format!("_{}", v))
        .collect::<Vec<String>>();

    let mut init_equ_args = vec![KCBuffer("dst", CF64)];
    init_equ_args.extend(dvars.iter().map(|pde| KCBuffer(&pde.0, CF64)));
    init_equ_args.extend(noises_names.iter().map(|n| KCBuffer(&n, CF64)));
    init_equ_args.push(KCParam("t", CF64));
    let init_equ_args: Vec<SKernelConstructor> =
        init_equ_args.into_iter().map(|a| a.into()).collect();
    let mut constraint_args = vec![KCBuffer("dst", CF64)];
    constraint_args.extend(dvars.iter().map(|pde| KCBuffer(&pde.0, CF64)));
    constraint_args.extend(o_vars_names.iter().map(|n| KCBuffer(n, CF64)));
    constraint_args.extend(noises_names.iter().map(|n| KCBuffer(n, CF64)));
    let constraint_string_id = "constraint_".len();
    constraint_args.extend(
        pure_constraints
            .iter()
            .map(|(n, _)| KCBuffer(&n[constraint_string_id..], CF64)),
    );
    constraint_args.push(KCParam("t", CF64));
    constraint_args.push(KCParam("dt", CF64));
    constraint_args.push(KCParam("cdt", CF64));
    constraint_args.push(KCBuffer("__err", CF64));
    let constraint_args: Vec<SKernelConstructor> =
        constraint_args.into_iter().map(|a| a.into()).collect();

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

    for (constraint, pure) in &constraints {
        if !pure {
            dvars
                .iter()
                .find(|(n, _)| n == &constraint.name)
                .expect(&format!(
                    "There must be a PDE corresponding to constraint \"{}\"",
                    &constraint.name
                ));
        }
        let name = format!("constraint_{}", &constraint.name);
        h = h.create_kernel(gen_single_stage_kernel(
            &name,
            constraint_args.clone(),
            constraint.clone(),
        ));
    }

    let mut equation_kernels = vec![];
    for equ in equations {
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
        let kernel_name = format!("equ_{}_{}", &name, 1); //WARNING: this 1 is probably useless
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

    let mut init_file: HashMap<String, Vec<f64>> = match param.init_file {
        Some(InitFormat::HDF5 { file, paths_names }) => {
            if !Path::new(&file).exists() {
                panic!("Init hdf5 file note found: \"{}\".", file);
            }
            let mut hdf5 = ConcurrentHDF5::new(&file)
                .expect(&format!("Could not read hdf5 file \"{}\".", file));
            paths_names
                .into_iter()
                .map(|(p, n)| {
                    (
                        n.clone(),
                        hdf5.read_data(&p)
                            .expect(&format!(
                                "Could not read data \"{}\" at path \"{}\" in init file \"{}\".",
                                n, p, file
                            ))
                            .data,
                    )
                })
                .collect()
        }
        Some(InitFormat::YAML { file }) => serde_yaml::from_str(
            &std::fs::read_to_string(&format!("{}{}", upparent, file)).expect(&format!(
                "Could not find initial conditions file \"{}\".",
                &file
            )),
        )
        .expect("Could not parse Yaml format input file"),
        None => HashMap::new(),
    };

    let mut max = 0;
    let mut name_to_index = HashMap::new();
    let mut index = 0;
    let num_pdes = dvars.len();
    dvars.iter_mut().for_each(|i| {
        max = usize::max(max, i.1);
        name_to_index.insert(i.0.clone(), index);
        index += 1; // 2 + nb_stages;
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
                    &format!("tmp_swap_{}_k{}", &dvar.0, (i + 1)),
                    Len(F64(0.0), len * dvar.1),
                );
                h = h.add_buffer(
                    &format!("tmp_save_{}_k{}", &dvar.0, (i + 1)),
                    Len(F64(0.0), len * dvar.1),
                );
            }
            h = h.add_buffer(&format!("tmp_{}_tmp", &dvar.0), Len(F64(0.0), len * dvar.1));
        }
    }
    h = h.add_buffer("tmp_error", Len(F64(1.0), len));
    for (name, dim) in &pure_constraints {
        h = h.add_buffer(name, Len(F64(0.0), len * dim));
        max = usize::max(max, *dim);
    }
    if init_file.len() > 0 {
        eprintln!("Warning, there are initial conditions that are not used from initial_conditions_file: {:?}.", init_file.keys())
    }

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

    h = h.create_kernel(SKernel {
        name: "to_var".into(),
        args: vec![(&KCBuffer("src", CF64)).into()],
        src: "    src[x+x_size] = sqrt(src[x+x_size]-src[x]*src[x]);".into(),
        needed: vec![],
    });

    if implicit {
        let vars = dvars
            .iter()
            .filter_map(|(n, i)| {
                if n.starts_with("dvar_") {
                    Some((n[5..].to_string(), *i))
                } else {
                    None
                }
            })
            .collect::<Vec<(String, usize)>>();

        let mut implicit_args = vec![];
        let mut error_args_names = vec![];
        for (name, _) in &vars {
            error_args_names.push(
                (0..nb_stages)
                    .map(|s| format!("{}_k{}", name, s))
                    .chain((0..nb_stages).map(|s| format!("{}_fk{}", name, s)))
                    .collect::<Vec<_>>(),
            );
        }
        for i in 0..vars.len() {
            for s in 0..nb_stages {
                implicit_args.push(KCBuffer(&error_args_names[i][s], CF64));
            }
            for s in 0..nb_stages {
                implicit_args.push(KCBuffer(&error_args_names[i][nb_stages + s], CF64));
            }
        }
        implicit_args.push(KCBuffer("err", CF64));
        implicit_args.push(KCParam("e", CF64));

        let mut implicit_src = "    double tmp = ".to_string();
        let mut implicit_src_end = "".to_string();
        for i in 0..vars.len() {
            let name = &vars[i].0;
            let vect_dim = vars[i].1;
            for s in 0..nb_stages {
                for vi in 0..vect_dim {
                    let tmp = if vect_dim == 1 {
                        format!(
                            //"fabs({n}_fk{s}[x]-{n}_k{s}[x])*ifNaNInf(1/fmax(fabs({n}_fk{s}[x]),fabs({n}_k{s}[x])), 1)",
                            "fabs({n}_fk{s}[x]-{n}_k{s}[x])",
                            n = name,
                            s = s
                        )
                    } else {
                        format!(
                            //"fabs({n}_fk{s}[{vd}*x+{vi}]-{n}_k{s}[{vd}*x+{vi}])*ifNaNInf(1/fmax(fabs({n}_fk{s}[{vd}*x+{vi}]),fabs({n}_k{s}[{vd}*x+{vi}])), 1)",
                            "fabs({n}_fk{s}[{vd}*x+{vi}]-{n}_k{s}[{vd}*x+{vi}])",
                            n = name,
                            vd = vect_dim,
                            vi = vi,
                            s = s
                        )
                    };
                    if i == vars.len() - 1 && s == nb_stages - 1 {
                        implicit_src += &tmp;
                    } else {
                        implicit_src += &format!("fmaxNaNInf({},", tmp);
                        implicit_src_end += ")";
                    }
                }
            }
        }
        implicit_src += &implicit_src_end;
        implicit_src += ";\n    err[x] = (tmp>=e) || (tmp!=tmp) || (isinf(tmp));";
        h = h.create_kernel(SKernel {
            name: "implicit_error".into(),
            args: implicit_args.into_iter().map(|i| i.into()).collect(),
            src: implicit_src,
            needed: vec![],
        });

        h = h.create_kernel(SKernel {
            name: "reset_error".into(),
            args: vec![(&KCBuffer("dst", CF64)).into()],
            src: "    dst[x] = true;".into(),
            needed: vec![],
        });
        let mut propagate_str = "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = ".to_string();
        let d = dim.len();
        let dr = |i, p| {
            let mut a = [0, 0, 0, 0];
            a[i] = p;
            a
        };
        let coords = phy
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                if *p == 0.0 {
                    vec![]
                } else {
                    vec![dr(i, 1), dr(i, -1)]
                }
            })
            .chain([dr(0, 0)])
            .collect::<Vec<_>>();
        let mut ors = String::new();
        for b in boundaries {
            for c in &coords {
                ors += &format!(" || {}({},src)", b, coords_str(c, d, 1));
            }
        }
        propagate_str += &ors[4..];
        propagate_str += ";";
        h = h.create_kernel(SKernel {
            name: "propagate_error".into(),
            args: vec![
                (&KCBuffer("dst", CF64)).into(),
                (&KCBuffer("src", CF64)).into(),
            ],
            src: propagate_str,
            needed: vec![],
        });
    }

    if check {
        println!("{}", h.source_code());
        return Ok(None);
    };
    let mut dvars = dvars
        .into_iter()
        .map(|(n, d)| (n, d as u32))
        .collect::<Vec<_>>();

    if let Some(noises) = &mut param.noises {
        noises
            .iter_mut()
            .for_each(|n| n.set_name(format!("src{}", n.name())));
    }

    h = h.add_buffer("tmp", Len(F64(0.0), mlen * max));
    h = h.add_buffer("tmp2", Len(F64(0.0), mlen * max));
    h = h.add_buffer("tmp3", Len(F64(0.0), mlen * max));
    h = h.add_buffer("sum", Len(F64(0.0), mlen * max));
    //h = h.add_buffer("srcFFT", Len(F64_2([0.0, 0.0]), len * max));
    //h = h.add_buffer("tmpFFT", Len(F64_2([0.0, 0.0]), len * max));
    //h = h.add_buffer("dstFFT", Len(F64_2([0.0, 0.0]), len * max));
    //h = h.add_buffer("initFFT", Len(F64_2([0.0, 0.0]), len * max));

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

    for (name, dim) in &pure_constraints {
        dvars.push((name.clone(), *dim as u32));
        name_to_index.insert(name[constraint_string_id..].to_string(), index);
        index += 1;
    }

    for (name, stage) in eqpde_init_copy {
        handler.copy(&name, &stage)?;
    }

    let stage = (integrator, equation_kernels);

    let vars = Vars {
        t_0,
        t_max,
        dt_0,
        dt_max,
        dt_factor,
        dt_reset,
        max_iter,
        max_reset,
        nb_propagate,
        dim,
        dirs,
        len,
        dvars,
        stage,
        noises: param.noises,
        phy,
        parent,
    };
    let callbacks = param
        .actions
        .into_iter()
        .map(|(c, a)| (a.to_activation(), c.to_callback(&name_to_index, num_pdes)))
        .collect();

    Ok(Some((
        Simulation {
            handler,
            callbacks,
            vars,
        },
        hdf5_file,
    )))
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
        format!("    dst[__i] = {};\n", &eqd.expr[0])
    } else {
        let mut expr = String::new();
        for i in 0..len {
            expr += &format!("    dst[{i}+__i] = {};\n", &eqd.expr[i], i = i);
        }
        expr
    };
    let priors = eqd.priors.join("\n    ");
    SKernel {
        name: name.to_string(),
        args,
        src: format!("    uint __i = {};\n{}\n{}", id, priors, expr),
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
    mut symbols: String,
    mut consts: HashMap<String, String>,
    default_boundary: String,
    mut dpdes: Vec<DPDE>,
    dim: usize,
    global_dim: usize,
) -> (
    Vec<SFunction>,
    Vec<SPDE>,
    Vec<EqDescriptor>,
    Vec<EqDescriptor>,
    Vec<(EqDescriptor, bool)>, // (constraint, is it pure)
    usize,
) {
    symbols = symbols
        .lines()
        .map(|l| {
            if let Some(i) = l.find("//") {
                &l[0..i]
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let re = Regex::new(r"\b\w+\b").unwrap();
    let replace = |src: &str, consts: &HashMap<String, String>| {
        re.replace_all(src, |caps: &Captures| {
            consts.get(&caps[0]).unwrap_or(&caps[0].to_string()).clone()
        })
        .to_string()
    };

    let mut func = vec![];
    let mut init = vec![];

    let search_const = Regex::new(r"^\s*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_func = Regex::new(r"^\s*(\w+)\((.+?)\)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_pde = Regex::new(r"^\s*(\w+)'\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_e = Regex::new(r"^\s*(\w+)\|\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_constraint = Regex::new(r"^\s*(\w+)'c\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_eqpde = Regex::new(r"^\s*(\w+)'\|\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_betweenpde = Regex::new(r"^\s*(\w+)'(-?\d*)>\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_init = Regex::new(r"^\s*\*(\w+)\s+(:?)=\s*(.+?)\s*$").unwrap();
    let search_empty = Regex::new(r"^\s*$").unwrap();

    use crate::gpgpu::pde_parser::*;
    let mut lexer_idx = 0;

    for l in symbols.lines() {
        // search for pdes or expressions
        if let Some(caps) = search_pde
            .captures(l)
            .or(search_e.captures(l))
            .or(search_eqpde.captures(l))
            .or(search_betweenpde.captures(l))
            .or(search_constraint.captures(l))
        {
            let name = &caps[1];
            if name.starts_with("_") {
                panic!("names starting with un '_' are reserved, the name \"{}\" is invalide, please remove the leading underscores.", name);
            }
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
    symbols = format!(
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
    ghost({c}_w,_w_size,*u) := ghost{n}({p}w,w_size,u)
{symbols}",
        c = choice(["_x,", "_y,", "_z,"]),
        n = global_dim,
        p = choice(["x,", "y,", "z,"]),
        symbols = symbols
    );
    let mut pdes = vec![];
    let mut constraints = vec![];
    let mut equations = vec![];
    let mut nb_propagate = 0;

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
                "Parse error on line{}:\n|------------\n{}\n|------------\n",
                if $j == $nj {
                    format!(" {}", $j + 1)
                } else {
                    format!("s {}-{}", $j, $nj)
                },
                &$src,
            ));
            lexer_idx += parsed.funs.len();
            func.append(&mut parsed.funs);
            nb_propagate = nb_propagate.max(parsed.max_space_derivative_depth);
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
                        let $i = caps[2].parse::<i32>().unwrap();
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
            panic!("Line {} could not be parsed.", j + 1);
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
    let mut sorted_constraints = constraints.clone();
    sorted_constraints.sort_by_key(|EqDescriptor { name, .. }| name.clone());
    pdes.iter_mut().for_each(|i| {
        i.expr.iter_mut().for_each(|e| *e = replace(e, &consts));
        if let Ok(constraint_id) = sorted_constraints
            .binary_search_by_key(&i.dvar, |EqDescriptor { name, .. }| name.clone())
        {
            i.constraint = Some(format!(
                "constraint_{}",
                sorted_constraints[constraint_id].name
            ));
        }
    });
    let pure_constraint = |n: &str| {
        !(pdes.iter().fold(false, |acc, pde| acc || pde.dvar == n)
            || equations.iter().fold(false, |acc, eq| acc || eq.name == n))
    };
    let constraints = constraints
        .into_iter()
        .map(|mut i| {
            i.expr.iter_mut().for_each(|e| *e = replace(e, &consts));
            let pure = pure_constraint(&i.name);
            (i, pure)
        })
        .collect::<Vec<_>>();
    equations
        .iter_mut()
        .for_each(|i| i.expr.iter_mut().for_each(|e| *e = replace(e, &consts)));

    (func, pdes, init, equations, constraints, nb_propagate)
}
