use crate::fuse::fuse;
use clap::{App, Arg, SubCommand};
use guess::*;
use ocl::error::Error::OclCore;
use ocl::OclCoreError::ProgramBuild;
use regex::Regex;
use simulation::NumType::*;

fn main() -> crate::gpgpu::Result<()> {
    let matches = App::new("guess")
                          .version("0.1")
                          .author("Nathan Touroux <touroux.nathan@gmail.com>")
                          .about("PDE simulation")
                          .subcommand(SubCommand::with_name("run")
                              .about("Run the PDE simulation")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file(s) to use")
                                  .max_values(1<<20)
                                  .required(true)))
                          .subcommand(SubCommand::with_name("check")
                              .about("Check that the parameter file is sintaxically correct and print the resulting opencl code.")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file to use")
                                  .required(true)))
                          .subcommand(SubCommand::with_name("fuse")
                              .about("fuse the results from the simulations named INPUT_<num> (where <num> might be any number).")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the simulations folde name to use")
                                  .required(true)))

                          .get_matches();

    let var = "GUESS_RUN";
    let rarg = Regex::new(r"((?:.*/)?[^:]+)(?::(\d+)?(=)?(\d+))?(#\d+)?").unwrap();
    let extract = |param: &str| {
        let caps = rarg.captures(param).expect("Input args should be in the form \"file_name:number\" where \":number\" is optional (\"file_name\" may contains \":\").");
        let param: String = caps[1].into();
        let num = if let Some(num) = caps.get(4).and_then(|n| Some(n.as_str())) {
            let num = num
                .parse::<usize>()
                .expect(&format!("Could not convert \"{}\" to number.", &caps[4]));
            if caps.get(3).map_or("", |c| c.as_str()) == "=" {
                if let Some((start, num)) = caps.get(2).and_then(|n| {
                    n.as_str()
                        .parse::<usize>()
                        .ok()
                        .and_then(|n| Some((num, n)))
                }) {
                    Multiple(num, start)
                } else {
                    Single(num)
                }
            } else {
                if let Some(dec) = caps.get(2).and_then(|i| {
                    Some(
                        i.as_str()
                            .parse::<usize>()
                            .expect(&format!("Could not convert \"{}\" to number.", &caps[2])),
                    )
                }) {
                    Multiple(dec * 10 + num, 0)
                } else {
                    Multiple(num, 0)
                }
            }
        } else {
            NoNum
        };
        let total_to_fuse = if let Some(tot) = caps.get(5).and_then(|n| Some(n.as_str())) {
            Some(
                tot[1..]
                    .parse::<usize>()
                    .expect(&format!("Could not convert \"{}\" to number.", &caps[4])),
            )
        } else {
            None
        };
        (param, num, total_to_fuse)
    };
    let run_sim = |params: Vec<&str>| -> crate::gpgpu::Result<()> {
        for param in params {
            let (param, num, total_to_fuse) = extract(param);
            match Simulation::from_param(&param, num, total_to_fuse, false) {
                Err(OclCore(ProgramBuild(log))) => {
                    eprintln!(
                        "Error while processing param \"{}\":\nBuild error:\n{}",
                        param, log
                    );
                }
                Err(e) => {
                    eprintln!("Error while processing param \"{}\":\n{:?}", param, e);
                }
                _ => {}
            }
        }

        Ok(())
    };
    if let Some(run) = matches.subcommand_matches("run") {
        run_sim(run.values_of("INPUT").unwrap().map(|i| i.into()).collect())?;
    } else if let Some(check) = matches.subcommand_matches("check") {
        Simulation::from_param(&check.value_of("INPUT").unwrap(), NoNum, None, true)?;
    } else if let Some(fuseargs) = matches.subcommand_matches("fuse") {
        fuse(vec![fuseargs.value_of("INPUT").unwrap().to_string()]);
    } else {
        let params = std::env::var(var).expect(&format!("The environment variable \"{}\" must be set as subcommand \"run\" would be set when no subcommands are given.", var));
        run_sim(params.split(" ").collect::<Vec<_>>())?;
    }

    Ok(())
}
