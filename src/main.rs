use ushf::*;
use clap::{Arg, App, SubCommand};

fn main() -> gpgpu::Result<()> {
    let matches = App::new("ushf")
                          .version("0.1")
                          .author("Nathan Touroux <touroux.nathan@gmail.com>")
                          .about("PDE simulation")
                          .subcommand(SubCommand::with_name("run")
                                        .arg(Arg::with_name("INPUT")
                                        .help("Sets the input file(s) to use")
                                        .max_values(1<<20)
                                        .required(true)))
                          .get_matches();
    if let Some(run) = matches.subcommand_matches("run") {
        for param in run.values_of("INPUT").unwrap() {
            let mut simulation = Simulation::from_param(&param)?;
            simulation.run()?;
        }
    }

    Ok(())
}
