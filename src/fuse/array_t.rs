use std::fmt;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::{Add, BitXor};

type ValueT = Vec<f64>;

#[derive(Debug, Clone)]
pub enum ArrayT {
    Values(ValueT),
    WithCoord(f64, ValueT),
}

impl ArrayT {
    pub fn is_nan(&self) -> bool {
        match self {
            ArrayT::Values(v) | ArrayT::WithCoord(_, v) => {
                v.iter().fold(false, |acc, v| acc || v.is_nan())
            }
        }
    }
}

impl Add<ArrayT> for ArrayT {
    type Output = ArrayT;
    fn add(self, rhs: ArrayT) -> Self::Output {
        match (self, rhs) {
            (ArrayT::Values(v1), ArrayT::Values(v2)) => {
                if v1.len() != v2.len() {
                    panic!("The lenght of arrays inside ArrayT must be of same lenght when added.")
                }
                ArrayT::Values(v1.iter().zip(v2.iter()).map(|(i1, i2)| i1 + i2).collect())
            }
            (ArrayT::WithCoord(c1, v1), ArrayT::WithCoord(c2, v2)) => {
                if c1 != c2 {
                    panic!("Coordinates must be the same when adding WithCoord variant of ArrayT.")
                }
                if v1.len() != v2.len() {
                    panic!("The lenght of arrays inside ArrayT must be of same lenght when added.")
                }
                ArrayT::WithCoord(
                    c1,
                    v1.iter().zip(v2.iter()).map(|(i1, i2)| i1 + i2).collect(),
                )
            }
            _ => panic!("ArrayT can be added only if they are of the same variant."),
        }
    }
}

impl fmt::Display for ArrayT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayT::Values(v) => write!(
                f,
                "{}",
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            ),
            ArrayT::WithCoord(c, v) => write!(
                f,
                "{}:{}",
                c.to_string(),
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
            ),
        }
    }
}
impl ArrayT {
    pub fn len(&self) -> usize {
        match self {
            ArrayT::Values(v) | ArrayT::WithCoord(_, v) => v.len(),
        }
    }
}
impl Mul<usize> for ArrayT {
    type Output = ArrayT;
    fn mul(self, n: usize) -> Self::Output {
        match self {
            ArrayT::Values(v) => ArrayT::Values(v.into_iter().map(|i| i * n as f64).collect()),
            ArrayT::WithCoord(c, v) => {
                ArrayT::WithCoord(c, v.into_iter().map(|i| i * n as f64).collect())
            }
        }
    }
}
impl BitXor<usize> for ArrayT {
    type Output = ArrayT;
    fn bitxor(self, n: usize) -> Self::Output {
        match self {
            ArrayT::Values(v) => {
                ArrayT::Values(v.into_iter().map(|i| f64::powf(i, n as f64)).collect())
            }
            ArrayT::WithCoord(c, v) => {
                ArrayT::WithCoord(c, v.into_iter().map(|i| f64::powf(i, n as f64)).collect())
            }
        }
    }
}
impl DivAssign<usize> for ArrayT {
    fn div_assign(&mut self, n: usize) {
        match self {
            ArrayT::Values(v) => {
                for i in 0..v.len() {
                    v[i] /= n as f64
                }
            }
            ArrayT::WithCoord(_, v) => {
                for i in 0..v.len() {
                    v[i] /= n as f64
                }
            }
        }
    }
}
