use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug,Clone)]
pub enum Repetition {
    At(f64),
    Every(f64),
    Interval{from: f64, to: f64, every: f64},
    TotalInterval{from: f64, to: f64, total: f64},
}
pub use Repetition::*;

pub type ActivationCallback = Box<dyn FnMut(f64) -> bool>;

impl Repetition {
    pub fn to_activation(self) -> ActivationCallback {
        match self {
            At(at) => {
                let mut done = false;
                Box::new(move |t| if !done && t >= at { done = true; true } else { false })
            },
            Every(every) => {
                let mut pred = f64::NEG_INFINITY;
                Box::new(move |t| if t >= pred + every { pred = t; true } else { false })
            },
            Interval{from,to,every} => {
                let mut pred = f64::NEG_INFINITY;
                Box::new(move |t| if from<=t && t<=to && t >= pred+every { pred = t; true } else { false })
            },
            TotalInterval{from,to,total} => {
                let every = (to-from)/total;
                let mut pred = f64::NEG_INFINITY;
                Box::new(move |t| if from<=t && t<=to && t >= pred+every { pred = t; true } else { false })
            },
        }
    }
}
