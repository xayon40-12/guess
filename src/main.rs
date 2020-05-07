use ushf::*;
use clap::{Arg, App, SubCommand};
use regex::Regex;
use crate::simulation::NumType::*;

fn main() -> gpgpu::Result<()> {
    let matches = App::new("ushf")
                          .version("0.1")
                          .author("Nathan Touroux <touroux.nathan@gmail.com>")
                          .about("PDE simulation")
                          .subcommand(SubCommand::with_name("run")
                              .about("Run the PDE simulation")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file(s) to use")
                                  .max_values(1<<20)
                                  .required(true)))
                          .subcommand(SubCommand::with_name("sub")
                              .about("Run the PDE simulation with qsub")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file(s) to use")
                                  .max_values(1<<20)
                                  .required(true)))
                          .subcommand(SubCommand::with_name("check")
                              .about("Check that the parameter file is sintaxically correct and print the resulting opencl code.")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file to use")
                                  .required(true)))
                          .subcommand(SubCommand::with_name("plot")
                              .about("Generate plots of the observables.")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the input folder where the observables data are.")))
                          .subcommand(SubCommand::with_name("fuse")
                              .about("Fuse the numbered folders of multiple simulation that come from the same parameter file.")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the name without number extension of the folders where the observables data are that are wanted to be averaged together.")))

                          .get_matches();

    let var = "USHF_RUN";
    let subcmd = |val| std::process::Command::new("qsub")
        .args(&[
            "-P","P_nantheo",
            "-l","GPU=1",
            "-l","GPUtype=V100",
            "-q","mc_gpu_long",
            "-pe","multicores_gpu","4",
            "-v",&format!("{}=\"{}\" ushf", var, val)
        ])
        .spawn()
        .expect("Could not launch qsub.");
    let rarg = Regex::new(r"([^:]*)(?::(=)?(.*))?").unwrap();
    let extract = |param: &str| {
            let caps = rarg.captures(param).expect("Input args should be in the form \"file_name:number\" where \":number\" is optional.");
            let param: String = caps[1].into();
            let num = if let Some(num) = caps.get(3).and_then(|n| Some(n.as_str())) {
                let num = num.parse::<usize>()
                .expect(&format!("Could not convert \"{}\" to number.",&caps[3]));
                if caps.get(2).map_or("", |c| c.as_str()) == "=" {
                    Single(num)
                } else {
                    Multiple(num)
                }
            } else {
                NoNum
            };
            (param,num)
    };
    let run_sim = |params: Vec<&str>| -> gpgpu::Result<()> {
        for param in params {
            let (param,num) = extract(param);
            Simulation::from_param(&param, num, false)?;
        }

        Ok(())
    };
    if let Some(run) = matches.subcommand_matches("run") {
        run_sim(run.values_of("INPUT").unwrap().map(|i| i.into()).collect())?;
    } else if let Some(check) = matches.subcommand_matches("check") {
        Simulation::from_param(&check.value_of("INPUT").unwrap(), NoNum, true)?;
    } else if let Some(sub) = matches.subcommand_matches("sub") {
        let param = sub.values_of("INPUT").unwrap().map(|i| {
            let (param,num) = extract(i);
            (std::fs::canonicalize(&param)
                    .expect(&format!("Could not find full path of file \"{}\"", param))
                    .to_str()
                    .unwrap()
                    .to_string(),
                    num
            )

        });
        for p in param {
            match p.1 {
                Multiple(num) => {
                    for i in 0..num {
                        subcmd(format!("{}:={}",p.0,i));
                    }
                },
                Single(num) => {
                    subcmd(format!("{}:={}",p.0,num));
                },
                NoNum => {
                    subcmd(p.0);
                },
            }
        }
    } else if let Some(_plot) = matches.subcommand_matches("plot") {
        panic!("The \"plot\" subcommand is not handled yet.")
    } else if let Some(_fuse) = matches.subcommand_matches("fuse") {
        panic!("The \"fuse\" subcommand is not handled yet.")
    } else {
        let params = std::env::var(var).expect(&format!("The environment variable \"{}\" must be set as subcommand \"run\" would be set when no subcommands are given.", var));
        run_sim(params.split(" ").collect::<Vec<_>>())?;

    }

    Ok(())
}
