use ushf::*;

fn main() -> gpgpu::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(),2);
    let simulation = Simulation::from_param(&args[1])?;

    Ok(())
}
