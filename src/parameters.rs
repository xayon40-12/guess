use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug)]
pub enum Counting {
    Count(u64),
    Fm(f64)
}
pub use Counting::*;

#[derive(Deserialize,Serialize,Debug)]
pub enum Repetition {
    At(Counting),
    Every(Counting)
}
pub use Repetition::*;

#[derive(Deserialize,Serialize,Debug)]
pub enum Action {
    Update(Repetition),
}
pub use Action::*;

#[derive(Deserialize,Serialize,Debug)]
pub struct Param {
    pub data_files: Vec<String>,
    pub eq_diff: String,
    pub actions: Vec<Action>,
}


