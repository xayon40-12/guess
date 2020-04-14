use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug)]
pub enum Counting {
    Count(u64),
    Fm(f64)
}
pub use Counting::*;

impl Counting {
    pub fn convert(&self, dt: f64) -> u64 {
        match self {
            Count(u) => *u,
            Fm(f) => (f/dt) as _
        }
    }
}

#[derive(Deserialize,Serialize,Debug)]
pub enum Repetition {
    At(Counting),
    Every(Counting)
}
pub use Repetition::*;

pub type ActivationCallback = Box<dyn Fn(u64) -> bool>;

impl Repetition {
    pub fn to_activation(&self, dt: f64) -> ActivationCallback {
        match self {
            At(c) => {
                let c: u64 = c.convert(dt);
                Box::new(move |count| count == c)
            },
            Every(c) => {
                let c: u64 = c.convert(dt);
                Box::new(move |count| count % c == 0)
            }
        }
    }
}

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

#[derive(Deserialize,Serialize,Debug)]
pub struct Param {
    pub data_files: Vec<String>,
    pub actions: Vec<Action>,
    pub symbols: SymbolsType,
}

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Update(Repetition),
}
pub use Action::*;

pub type Callback = (ActivationCallback,Box<dyn Fn(gpgpu::Handler,u64) -> gpgpu::Result<()>>);

impl Action {
    pub fn to_callback(&self, dt: f64) -> Callback {
        match self {
            Update(rep) => (rep.to_activation(dt), Box::new(|h,t| {

                Ok(())
            })),
        }
    }
}
