use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug)]
pub enum Counting {
    Count(usize),
    Fm(f64)
}
pub use Counting::*;

impl Counting {
    pub fn convert(&self, dt: f64) -> usize {
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

pub type ActivationCallback = Box<dyn Fn(usize) -> bool>;

impl Repetition {
    pub fn to_activation(&self, dt: f64) -> ActivationCallback {
        match self {
            At(c) => {
                let c: usize = c.convert(dt);
                Box::new(move |count| count == c)
            },
            Every(c) => {
                let c: usize = c.convert(dt);
                Box::new(move |count| count % c == 0)
            }
        }
    }
}
