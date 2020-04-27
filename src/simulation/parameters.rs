use serde::{Deserialize,Serialize};
use crate::simulation::parameters::actions::Action;
use crate::simulation::parameters::activations::Repetition;
use gpgpu::Dim;
use gpgpu::DimDir;

pub mod actions;
pub use actions::Callback;
pub mod activations;
pub use activations::ActivationCallback;
pub mod symbols;
pub use symbols::SymbolsTypes;
pub mod integrators;
pub use integrators::Integrator;

#[derive(Deserialize,Serialize,Debug)]
pub struct Param {
    pub data_files: Vec<String>,
    pub actions: Vec<(Action,Repetition)>,
    pub symbols: Vec<SymbolsTypes>,
    pub config: Config,
    pub integrators: Vec<Integrator>,
}

#[derive(Deserialize,Serialize,Debug)]
pub struct Config {
    pub t_max: f64,
    pub dim: Dim,
    pub dirs: Vec<DimDir>,
}

