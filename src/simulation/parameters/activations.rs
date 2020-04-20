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
    Every(Counting),
    Interval{from: Counting, to: Counting, by: Counting},
    TotalInterval{from: Counting, to: Counting, total: usize},
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
            },
            Interval{from,to,by} => {
                let (f,t,b): (usize,usize,usize) = (from.convert(dt),to.convert(dt),by.convert(dt));
                Box::new(move |c| f<=c && c<=t && c%b == 0)
            },
            TotalInterval{from,to,total} => {
                let (f,t,tot): (usize,usize,usize) = (from.convert(dt),to.convert(dt),*total);
                let b = (t-f)/tot;
                Box::new(move |c| f<=c && c<=t && c%b == 0)
            },
        }
    }
}
