use serde::{Deserialize,Serialize};
use gpgpu::integrators::SPDE;

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum SymbolsTypes {
    Constant{ name: String, value: f64 },
    Function{ name: String, args: Vec<String>, indx_args: Option<Vec<String>>, src: String },
    PDEs(Vec<SPDE>),
}
