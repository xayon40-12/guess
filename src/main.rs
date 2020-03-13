use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize,Debug)]
struct Param {
    vals: BTreeMap<String,f64>
}

fn around(map: BTreeMap<String,f64>, val: f64) -> (Option<(f64,f64)>,Option<(f64,f64)>) {
    let a = map.range(val.to_string()..).next().and_then(|(a,&b)| Some((a.parse().unwrap(),b)));
    let b = map.range(..val.to_string()).next_back().and_then(|(a,&b)| Some((a.parse().unwrap(),b)));
    (a,b)
}


fn main() {
    let param: BTreeMap<String,f64> = ron::de::from_str(include_str!("param.ron")).unwrap();


    println!("{:?}", around(param, 3.0));
}
