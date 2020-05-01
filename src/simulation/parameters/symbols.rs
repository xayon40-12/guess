use serde::{Deserialize,Serialize};
use gpgpu::integrators::SPDE;

#[derive(Deserialize,Serialize,Debug,Clone)]
pub struct Init {
    pub name: String,
    pub expr: Vec<String>,
}

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum SymbolsTypes {
    Constant{ name: String, value: f64 },
    Function{ name: String, args: Vec<(String, bool)>, src: String },
    PDEs(Vec<SPDE>),
    Init(Vec<Init>),//WARNING like for pdes, they will be computed in reverse order (ORDER MATERS)
}
