use clap::{App, Arg, SubCommand};
use regex::Regex;
use simulation::NumType::*;
use ushf_run::*;

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
                          .subcommand(SubCommand::with_name("check")
                              .about("Check that the parameter file is sintaxically correct and print the resulting opencl code.")
                              .arg(Arg::with_name("INPUT")
                                  .help("Sets the parameter file to use")
                                  .required(true)))
                          .get_matches();

    let var = "USHF_RUN";
    let rarg = Regex::new(r"((?:.*/)?[^:]+)(?::(=?)(\d+))?").unwrap();
    let extract = |param: &str| {
        let caps = rarg.captures(param).expect("Input args should be in the form \"file_name:number\" where \":number\" is optional (\"file_name\" may contains \":\").");
        let param: String = caps[1].into();
        let num = if let Some(num) = caps.get(3).and_then(|n| Some(n.as_str())) {
            let num = num
                .parse::<usize>()
                .expect(&format!("Could not convert \"{}\" to number.", &caps[3]));
            if caps.get(2).map_or("", |c| c.as_str()) == "=" {
                Single(num)
            } else {
                Multiple(num)
            }
        } else {
            NoNum
        };
        (param, num)
    };
    let run_sim = |params: Vec<&str>| -> gpgpu::Result<()> {
        for param in params {
            let (param, num) = extract(param);
            Simulation::from_param(&param, num, false)?;
        }

        Ok(())
    };
    if let Some(run) = matches.subcommand_matches("run") {
        run_sim(run.values_of("INPUT").unwrap().map(|i| i.into()).collect())?;
    } else if let Some(check) = matches.subcommand_matches("check") {
        Simulation::from_param(&check.value_of("INPUT").unwrap(), NoNum, true)?;
    } else {
        let params = std::env::var(var).expect(&format!("The environment variable \"{}\" must be set as subcommand \"run\" would be set when no subcommands are given.", var));
        run_sim(params.split(" ").collect::<Vec<_>>())?;
    }

    Ok(())
}
