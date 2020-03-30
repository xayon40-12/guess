use ushf::*;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(),2);
    let param: Param = serde_yaml::from_str(&std::fs::read_to_string(&args[1]).expect(&format!("Could not find parameter file \"{}\".", &args[1]))).unwrap();
    println!("param:\n{:?}", param);
}
