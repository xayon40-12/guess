use serde::{Deserialize,Serialize};
use crate::simulation::parameters::actions::Action;
use crate::simulation::parameters::activations::Repetition;
use gpgpu::DimDir;

pub mod actions;
pub use actions::Callback;
pub mod activations;
pub use activations::ActivationCallback;
pub mod symbols;
pub use symbols::SymbolsTypes;

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum Integrator {
    Euler{dt: f64},
    QSS,
}

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum Noises {
    Uniform{name: String, dim: Option<usize>},
    Normal{name: String, dim: Option<usize>},
}

impl Noises {
    pub fn name(&self) -> String {
        use Noises::*;
        match self {
            Uniform{name,..} | Normal{name,..} => name.clone(),
        }
    }
    pub fn set_name(&mut self, new_name: String) {
        use Noises::*;
        match self {
            Uniform{name,..} | Normal{name,..} => *name = new_name,
        }
    }
}

#[derive(Deserialize,Serialize,Debug,Clone)]
pub struct Param {
    pub data_files: Option<Vec<String>>,
    pub actions: Vec<(Action,Repetition)>,
    pub noises: Option<Vec<Noises>>,
    pub symbols: String,
    pub config: Config,
    pub integrator: Integrator,
    pub initial_conditions_file: Option<String>,
}

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum DimPhy {
    D1((usize,f64)),
    D2((usize,f64),(usize,f64)),
    D3((usize,f64),(usize,f64),(usize,f64)),
}

impl From<DimPhy> for ([usize;3],[f64;3]) {
    fn from(d: DimPhy) -> Self {
        match d {
            DimPhy::D1((u,f)) => ([u,1,1],[f,0.0,0.0]),
            DimPhy::D2((u0,f0),(u1,f1)) => ([u0,u1,1],[f0,f1,0.0]),
            DimPhy::D3((u0,f0),(u1,f1),(u2,f2)) => ([u0,u1,u2],[f0,f1,f2]),
        }
    }
}

#[derive(Deserialize,Serialize,Debug,Clone)]
pub struct Config {
    pub t_max: f64,
    pub dim: DimPhy,
    pub dirs: Vec<DimDir>,
}

