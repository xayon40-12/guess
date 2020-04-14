use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug)]
pub enum SymbolsType {
    Rust(Vec<RustSymbols>),
    Sympy(String)
}

#[derive(Deserialize,Serialize,Debug)]
pub enum RustSymbols {
    Function(String),
    Main(String),
}
