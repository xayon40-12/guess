use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum Repetition {
    At(f64),
    Every(f64),
    Interval { from: f64, to: f64, every: f64 },
    TotalInterval { from: f64, to: f64, total: f64 },
}
pub use Repetition::*;

pub type ActivationCallback = Box<dyn FnMut(f64) -> f64>;

impl Repetition {
    pub fn to_activation(self) -> ActivationCallback {
        match self {
            At(at) => Box::new(move |t| at - t),
            Every(every) => {
                let mut next = 0.0;
                Box::new(move |t| {
                    let d = next - t;
                    if t >= next {
                        next = t - (t - next) % every + every;
                    }
                    d
                })
            }
            Interval { from, to, every } => {
                let mut next = from;
                Box::new(move |t| {
                    let d = next - t;
                    if t <= to && t >= next {
                        next = t - (t - next) % every + every;
                    }
                    d
                })
            }
            TotalInterval { from, to, total } => {
                let total = if total < 3.0 { 2.0 } else { total - 1.0 };
                let every = (to - from) / total;
                let mut next = from;
                Box::new(move |t| {
                    let d = next - t;
                    if t <= to && t >= next {
                        next = t - (t - next) % every + every;
                    }
                    d
                })
            }
        }
    }
}
