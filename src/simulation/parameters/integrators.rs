use serde::{Deserialize,Serialize};

#[derive(Deserialize,Serialize,Debug)]
pub enum Integrator {
    Euler,
    QSS,
}
