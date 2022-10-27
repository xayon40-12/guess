use crate::gpgpu::functions::SFunction;
use crate::gpgpu::pde_parser::pde_ir::SPDETokens;
use serde::{Deserialize, Serialize};
use std::ops::{Add, BitXor, Div, Mul, Sub};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexerComp {
    pub token: SPDETokens,
    pub funs: Vec<SFunction>,
    pub max_space_derivative_depth: usize,
}

impl LexerComp {
    pub fn map<T: FnOnce(SPDETokens) -> SPDETokens>(self, f: T) -> LexerComp {
        LexerComp {
            token: f(self.token),
            funs: self.funs,
            max_space_derivative_depth: self.max_space_derivative_depth,
        }
    }

    pub fn bind<T: FnOnce(SPDETokens) -> LexerComp>(mut self, f: T) -> LexerComp {
        let mut res = f(self.token);
        self.funs.append(&mut res.funs); // conserve order of function creation
        res.funs = self.funs;
        res.max_space_derivative_depth += self.max_space_derivative_depth;
        res
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compacted {
    tokens: Vec<SPDETokens>,
    funs: Vec<SFunction>,
    max_space_derivative_depth: usize,
}

impl Compacted {
    pub fn empty() -> Compacted {
        Compacted {
            tokens: vec![],
            funs: vec![],
            max_space_derivative_depth: 0,
        }
    }

    pub fn map<T: FnOnce(Vec<SPDETokens>) -> SPDETokens>(self, f: T) -> LexerComp {
        LexerComp {
            token: f(self.tokens),
            funs: self.funs,
            max_space_derivative_depth: self.max_space_derivative_depth,
        }
    }

    pub fn bind<T: FnOnce(Vec<SPDETokens>) -> LexerComp>(mut self, f: T) -> LexerComp {
        let mut res = f(self.tokens);
        self.funs.append(&mut res.funs);
        res.funs = self.funs;
        res.max_space_derivative_depth += self.max_space_derivative_depth;
        res
    }
}

pub fn compact(tab: Vec<LexerComp>) -> Compacted {
    tab.into_iter().fold(Compacted::empty(), |mut acc, mut i| {
        acc.tokens.push(i.token);
        acc.funs.append(&mut i.funs);
        acc.max_space_derivative_depth = acc
            .max_space_derivative_depth
            .max(i.max_space_derivative_depth);
        acc
    })
}

impl<T: Into<SPDETokens>> From<T> for LexerComp {
    fn from(pde: T) -> Self {
        LexerComp {
            token: pde.into(),
            funs: vec![],
            max_space_derivative_depth: 0,
        }
    }
}

macro_rules! op {
    ($name:ident|$fun:ident $op:tt) => {
impl $name for LexerComp {
    type Output = Self;
    fn $fun(self, r: Self) -> Self {
        let LexerComp {
            token: lt,
            funs: mut lf,
            max_space_derivative_depth: lm,
        } = self;
        let LexerComp {
            token: rt,
            funs: mut rf,
            max_space_derivative_depth: rm,
        } = r;
        lf.append(&mut rf);

        LexerComp {
            token: lt $op rt,
            funs: lf,
            max_space_derivative_depth: rm.max(lm),
        }
    }
}
    };
}

op! {Add|add +}
op! {Sub|sub -}
op! {Mul|mul *}
op! {Div|div /}
op! {BitXor|bitxor ^}
