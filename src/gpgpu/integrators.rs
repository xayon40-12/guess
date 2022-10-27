use crate::gpgpu::algorithms::ReduceParam;
use crate::gpgpu::descriptors::{ConstructorTypes, Types};
use crate::gpgpu::descriptors::{ConstructorTypes::*, KernelArg::*, KernelConstructor::*};
use crate::gpgpu::dim::{
    Dim::{self, *},
    DimDir,
};
use crate::gpgpu::kernels::Kernel;
use crate::gpgpu::Handler;
use crate::{
    gpgpu::algorithms::{
        AlgorithmParam, SAlgorithm,
        SNeeded::{self, *},
    },
    simulation::Integrator,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum STEP {
    PDE,
    EQPDE,
    BETWEENPDE(i32),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SPDE {
    pub dvar: String,
    pub expr: Vec<String>, //one String for each dimension of the vectorial pde
    pub priors: Vec<String>,
    pub step: STEP,
    pub constraint: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegratorParam {
    pub swap: usize,
    pub t: f64,
    pub t_name: String,
    pub dt: f64,
    pub dt_max: f64,
    pub dt_factor: f64,
    pub dt_reset: f64,
    pub dt_name: String,
    pub cdt_name: String, // current time step during a RungeKutta stages (so c_i*dt)
    pub args: Vec<(String, Types)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scheme {
    pub aij: Vec<Vec<f64>>,
    pub bj: Vec<f64>,
    pub bjs: Option<Vec<f64>>,
}

impl Scheme {
    pub fn check(&self) -> bool {
        let l = self.aij.len();
        for c in &self.aij {
            if c.len() != l {
                return false;
            }
        }
        if let Some(bjs) = &self.bjs {
            if bjs.len() != l {
                return false;
            }
        }
        self.bj.len() == l
    }
}

// Each PDE must be first order in time. A higher order PDE can be cut in multiple first order PDE.
// Example: d2u/dt2 + du/dt = u   =>   du/dt = z, dz/dt = u.
// It is why the parameter pdes is a Vec.
pub type CreatePDE = fn(
    &str,
    Integrator,
    Vec<SPDE>,
    Option<Vec<String>>,
    Vec<(String, ConstructorTypes)>,
) -> SAlgorithm;
pub fn create_euler_pde(
    name: &str,
    integrator: Integrator,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        integrator,
        Scheme {
            aij: vec![vec![0.0]],
            bj: vec![1.0],
            bjs: None,
        },
    )
}
pub fn create_projector_corrector_pde(
    name: &str,
    integrator: Integrator,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        integrator,
        Scheme {
            aij: vec![vec![0.0, 0.0], vec![0.5, 0.0]],
            bj: vec![0.0, 1.0],
            bjs: None,
        },
    )
}
pub fn create_rk4_pde(
    name: &str,
    integrator: Integrator,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        integrator,
        Scheme {
            aij: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.5, 0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            bj: vec![1. / 6., 1. / 3., 1. / 3., 1. / 6.],
            bjs: None,
        },
    )
}

pub fn create_implicit_radau_pde(
    name: &str,
    integrator: Integrator,
    pdes: Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
) -> SAlgorithm {
    multistages_algorithm(
        name,
        &pdes,
        needed_buffers,
        params,
        integrator,
        Scheme {
            aij: vec![vec![5.0 / 12.0, -1.0 / 12.0], vec![3.0 / 4.0, 1.0 / 4.0]],
            bj: vec![3.0 / 4.0, 1.0 / 4.0],
            bjs: Some(vec![1.0, 0.0]),
        },
    )
}

fn multistages_kernels(
    name: &str,
    pdes: &Vec<SPDE>,
    needed_buffers: &Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
    scheme: &Scheme,
    implicit: bool,
) -> Vec<SNeeded> {
    let mut args = vec![KCBuffer("dst", CF64)];
    args.extend(pdes.iter().map(|pde| KCBuffer(&pde.dvar, CF64)));
    if let Some(ns) = &needed_buffers {
        args.extend(ns.iter().map(|n| KCBuffer(&n, CF64)));
    }
    args.extend(params.iter().map(|t| KCParam(&t.0, t.1)));
    let mut needed = pdes
        .iter()
        .map(|d| {
            let mut id = "x+x_size*(y+y_size*z)".to_string();
            let len = d.expr.len();
            if len > 1 {
                id = format!("{}*({})", len, id);
            }
            let mut expr = String::new();
            for i in 0..len {
                expr += &format!("    dst[{i}+_i] = {};\n", &d.expr[i], i = i);
            }
            let priors = d.priors.join("\n    ");
            NewKernel(
                (&Kernel {
                    name: &format!("{}_{}", &name, &d.dvar),
                    args: args.clone(),
                    src: &format!("    {}\n    uint _i = {};\n{}", priors, id, expr),
                    needed: vec![],
                })
                    .into(),
            )
        })
        .collect::<Vec<_>>();
    for (i, v) in scheme.aij.iter().chain([&scheme.bj]).enumerate() {
        let mut args = vec![
            KCBuffer("dst", CF64),
            KCBuffer("src", CF64),
            KCParam("h", CF64),
        ];
        let argnames = (0..v.len())
            .map(|i| format!("src{}", i + 1))
            .collect::<Vec<_>>();
        let mut src = String::new();
        for (i, v) in v.iter().enumerate() {
            args.push(KCBuffer(&argnames[i], CF64));
            if *v != 0.0 {
                src = format!("{} + {}*{}[i]", src, v, &argnames[i]);
            }
        }
        if src.is_empty() {
            src = "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = src[i];".to_string();
        } else {
            src = format!(
                "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = src[i] + h*({});",
                &src[3..]
            );
        }
        let ssrc = "src".to_string();
        let eq_arg = if implicit {
            let eq_i = if i == v.len() { i - 1 } else { i }; // the last stage (using the bi) correspond to i=nb_stages=v.len(), so at that point, the last stage should be chosen for eq so nb_stages-1
            &argnames[eq_i]
        } else {
            if i == 0 {
                &ssrc
            } else {
                &argnames[i - 1] // WARNING: for some reason the values in argnames on the gpu are always 0
            }
        };
        let src_eq = format!(
            "    uint i = x+x_size*(y+y_size*z);\n    dst[i] = {}[i];",
            &eq_arg // TODO: optimize: use argnames[0] here and in multistages_algorithm for eq, then only alocate one buffer for them
        );

        needed.push(NewKernel(
            (&Kernel {
                name: &format!("{}_stage{}", name, i),
                args: args.clone(),
                src: &src,
                needed: vec![],
            })
                .into(),
        ));
        needed.push(NewKernel(
            (&Kernel {
                name: &format!("{}_stage{}_eq", name, i),
                args,
                src: &src_eq,
                needed: vec![],
            })
                .into(),
        ));
    }
    needed
}

fn multistages_algorithm(
    name: &str,
    pdes: &Vec<SPDE>,
    needed_buffers: Option<Vec<String>>,
    params: Vec<(String, ConstructorTypes)>,
    integrator: Integrator,
    scheme: Scheme,
) -> SAlgorithm {
    let name = name.to_string();
    let vars = pdes
        .iter()
        .map(|d| {
            (
                format!("{}_{}", &name, &d.dvar),
                d.dvar.clone(),
                d.expr.len(),
                d.step.clone(),
                d.constraint.clone(),
            )
        })
        .collect::<Vec<_>>();
    let mut vars_ranges = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, _, s, _))| match s {
            STEP::PDE | STEP::EQPDE | STEP::BETWEENPDE(0) => (0, i),
            STEP::BETWEENPDE(j) => (*j, i),
        })
        .collect::<Vec<_>>();
    vars_ranges.sort();
    let (_, vars_ranges) = vars_ranges
        .into_iter()
        .fold((0, vec![vec![]]), |(c, mut a), (j, i)| {
            if j == c {
                let l = a.len() - 1;
                a[l].push(i);
                (c, a)
            } else {
                a.push(vec![i]);
                (j, a)
            }
        });
    let nb_stages = scheme.aij.len(); // +1 for bij
    let nb_per_stages = 3 + 3 * nb_stages;
    let mut len = nb_per_stages * vars.len();
    let nb_pde_buffers = len;
    if let Some(ns) = &needed_buffers {
        len += ns.len();
    }
    if !scheme.check() {
        panic!("In a multistages algorithms the scheme must be well formed: the coefficient table 'aij' must be square along with the end sum vector 'bj' that must be the same length as the table. The second sum vuctor use to predict the time step must be the same size as well if provided.")
    }

    let tmpid = nb_per_stages - 1;
    let pre_constraint_id = nb_per_stages - 2;
    let (max_error, implicit) = match integrator {
        Integrator::Explicit { .. } => (0.0, false),
        Integrator::Implicit { er, .. } => (er, true),
    };
    let needed = multistages_kernels(&name, &pdes, &needed_buffers, params, &scheme, implicit);
    SAlgorithm {
        name: name.clone(),
        callback: std::rc::Rc::new(
            move |h: &mut Handler,
                  dim: Dim,
                  _dimdir: &[DimDir],
                  bufs: &[&str],
                  other: AlgorithmParam| {
                // bufs[0] = dst
                // bufs[1,2,...] = differential equation buffer holders in the same order as giver for
                // create_euler function
                // bufs[i] must write in bufs[i-1]
                let _dim: [usize; 3] = dim.into();
                let d = _dim.iter().fold(1, |a, i| a * i);
                if bufs.len() != len {
                    panic!(
                        "Multistages algorithm \"{}\" must be given {} buffer arguments, {} where given: {:?}.",
                        &name, &len, &bufs.len(), &bufs
                    );
                }
                let mut intprm = other
                .downcast_ref::<IntegratorParam>("There must be an Ref(&IntegratorParam) given as optional argument in Multistages integrator algorithm.").clone();
                let IntegratorParam {
                    mut swap,
                    t,
                    ref t_name,
                    mut dt,
                    dt_max,
                    dt_factor,
                    dt_reset,
                    ref dt_name,
                    ref cdt_name,
                    args: iargs,
                } = intprm.clone();
                let mut args = vec![BufArg("", ""); vars.len() + 1];
                if let Some(ns) = &needed_buffers {
                    let mut i = nb_pde_buffers;
                    for n in ns {
                        args.push(BufArg(&bufs[i], &n));
                        i += 1;
                    }
                }
                args.extend(iargs.iter().map(|i| Param(&i.0, i.1)));
                args.push(Param(t_name, t.into()));
                args.push(Param(dt_name, dt.into()));
                args.push(Param(cdt_name, 0.into()));
                let t_id = args.len() - 3;
                let dt_id = args.len() - 2;
                let cdt_id = args.len() - 1;
                let argnames = (0..nb_per_stages)
                    .map(|i| format!("src{}", i + 1))
                    .collect::<Vec<_>>();

                macro_rules! save {
                    () => {
                        for i in 0..vars.len() {
                            for j in 0..nb_stages {
                                h.copy(
                                    &bufs[nb_per_stages * i + swap * nb_stages + (j + 1)],
                                    &bufs[nb_per_stages * i + 2 * nb_stages + (j + 1)],
                                )?;
                            }
                        }
                    };
                }
                macro_rules! reset {
                    () => {
                        for i in 0..vars.len() {
                            for j in 0..nb_stages {
                                h.copy(
                                    &bufs[nb_per_stages * i + 2 * nb_stages + (j + 1)],
                                    &bufs[nb_per_stages * i + swap * nb_stages + (j + 1)],
                                )?;
                            }
                        }
                    };
                }

                let max_iter = 20;
                let max_error_iter = 2;
                let range = if implicit {
                    1..=max_error_iter * max_iter
                } else {
                    1..=1
                };
                macro_rules! stage {
                    ($s:ident,$r:ident,$coef:expr,$pred:ident) => {
                        let cdt = $coef.iter().fold(0.0, |a, i| a + i) * dt;
                        args[t_id] = Param(t_name, (t + cdt).into()); // increment time for next stage
                        args[dt_id] = Param(dt_name, dt.into()); // increment time for next stage
                        args[cdt_id] = Param(cdt_name, cdt.into()); // increment time for next stage
                        for &i in $r {
                            let constraint = &vars[i].4;
                            let dst_buf = &bufs[nb_per_stages * i + if constraint.is_some() { pre_constraint_id } else if $s == nb_stages { 0 } else { tmpid }]; //TODO: if constraint write in pre_constrained_id
                            let mut stage_args = vec![
                                BufArg(dst_buf,"dst",),
                                BufArg(&bufs[nb_per_stages * i ], "src"),
                                Param("h", dt.into()),
                            ];
                            stage_args.extend(
                                (0..nb_stages)
                                    .map(|j| BufArg(&bufs[nb_per_stages * i + swap*nb_stages + (j + 1)], &argnames[j])),
                            );
                            let step = &vars[i].3;
                            let stage_name = &match step {
                                STEP::PDE => format!("{}_stage{}", name, $s),
                                STEP::EQPDE | STEP::BETWEENPDE(_) => {
                                    format!("{}_stage{}_eq", name, $s)
                                }
                            };
                            $pred.push(i);
                            h.run_arg(stage_name, D1(d * vars[i].2), &stage_args)?;
                            // vars[i].2 correspond to the vectorial dim of the current pde
                        }

                        for &i in $r {
                            let constraint = &vars[i].4;
                            let dst_buf = &bufs[nb_per_stages * i + if $s == nb_stages { 0 } else { tmpid }];
                            if let Some(constraint_name) = constraint {
                                args[0] = BufArg(dst_buf, "dst");
                                for i in 0..vars.len() {
                                    let pos = if vars[i].4.is_some() && $pred.contains(&i) { pre_constraint_id } else if ($s == nb_stages && $pred.contains(&i)) || ($s == 0 && !$pred.contains(&i)) {
                                        0
                                    } else {
                                        tmpid
                                    };
                                    args[i+1] = BufArg(&bufs[nb_per_stages * i + pos], &vars[i].1);
                                }
                                h.run_arg(constraint_name, D1(d * vars[i].2), &args)?;
                            }
                        }
                    };
                }
                let mut done = false;
                for l in range {
                    let dst_swap = if implicit { 1 - swap } else { swap };
                    for s in 0..nb_stages {
                        let mut pred = vec![];
                        for r in &vars_ranges {
                            stage!(s, r, scheme.aij[s], pred);
                            for &i in r {
                                args[0] = BufArg(
                                    &bufs[nb_per_stages * i + dst_swap * nb_stages + (s + 1)],
                                    "dst",
                                );
                                for i in 0..vars.len() {
                                    let pos = if s == 0 && !pred.contains(&i) {
                                        0
                                    } else {
                                        tmpid
                                    };
                                    args[1 + i] =
                                        BufArg(&bufs[nb_per_stages * i + pos], &vars[i].1);
                                }
                                h.run_arg(&vars[i].0, dim, &args)?;
                            }
                        }
                    }
                    if implicit {
                        let mut error_args = vec![BufArg(&bufs[tmpid], "dst")];
                        let mut error_args_names = vec![];
                        for i in 0..vars.len() {
                            error_args_names.push(
                                (0..nb_stages)
                                    .map(|s| format!("{}_k{}", vars[i].1, s))
                                    .chain((0..nb_stages).map(|s| format!("{}_fk{}", vars[i].1, s)))
                                    .collect::<Vec<_>>(),
                            );
                        }
                        for i in 0..vars.len() {
                            for s in 0..nb_stages {
                                error_args.push(BufArg(
                                    &bufs[nb_per_stages * i + swap * nb_stages + (s + 1)],
                                    &error_args_names[i][s],
                                ));
                            }
                            for s in 0..nb_stages {
                                error_args.push(BufArg(
                                    &bufs[nb_per_stages * i + dst_swap * nb_stages + (s + 1)],
                                    &error_args_names[i][nb_stages + s],
                                ));
                            }
                        }
                        h.run_arg("implicit_error", D1(d), &error_args)?;
                        let ap = ReduceParam {
                            vect_dim: 1,
                            dst_size: None,
                            window: None,
                        };
                        let dst_sum = &bufs[tmpid];
                        h.run_algorithm(
                            "sum",
                            D1(d),
                            &[DimDir::X],
                            &[&bufs[tmpid], &bufs[pre_constraint_id], dst_sum],
                            AlgorithmParam::Ref(&ap),
                        )?;
                        let err: f64 = h.get_first(dst_sum)?.F64();
                        swap = 1 - swap;
                        if (err * nb_stages as f64) < max_error {
                            done = true;
                            break;
                        } else if l % max_iter == 0 {
                            dt *= dt_reset;
                            println!("reset");
                            reset!();
                        }
                    }
                }
                if implicit {
                    if !done {
                        panic!(
                            "Implicit scheme could not converge avec {} attempts",
                            max_error_iter
                        );
                    }
                    save!();
                }
                for r in &vars_ranges {
                    let mut pred = vec![];
                    stage!(nb_stages, r, scheme.bj, pred);
                }

                intprm.dt = dt_max.min(dt * dt_factor);
                intprm.t += intprm.dt;
                intprm.swap = swap;
                Ok(Some(Box::new(intprm)))
            },
        ),
        needed,
    }
}
