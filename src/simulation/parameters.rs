use serde::{Deserialize,Serialize};
use crate::simulation::parameters::actions::Action;
use crate::simulation::parameters::activations::Repetition;
use gpgpu::Dim;
use gpgpu::DimDir;

pub mod actions;
pub mod activations;
use activations::Counting;
pub mod symbols;
use symbols::SymbolsType;

#[derive(Deserialize,Serialize,Debug)]
pub struct Param {
    pub data_files: Vec<String>,
    pub actions: Vec<(Action,Repetition)>,
    pub symbols: SymbolsType,
    pub config: Config,
}

#[derive(Deserialize,Serialize,Debug)]
pub struct Config {
    pub max: Counting,
    pub dim: Dim,
    pub dirs: Vec<DimDir>,
}

