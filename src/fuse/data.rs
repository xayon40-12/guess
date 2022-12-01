use super::line::Line;
use std::fmt;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct Data {
    lines: Vec<Line>,
}

impl Data {
    pub fn new(folder: &str, filename: &str) -> Option<Data> {
        let lines = std::fs::read_to_string(&format!("{}/{}", folder, filename))
            .unwrap()
            .lines()
            .map(Line::from)
            .collect::<Vec<_>>();
        let is_nan = lines.iter().fold(false, |acc, l| acc || l.is_nan());
        if is_nan {
            None
        } else {
            Some(Data { lines })
        }
    }

    pub fn finish(&mut self) {
        self.lines.iter_mut().for_each(|l| l.mean())
    }
}

impl Add<Data> for Data {
    type Output = Data;
    fn add(self, rhs: Data) -> Self::Output {
        if self.lines.len() != rhs.lines.len() {
            panic!("Only similar Data can be added.")
        }
        Data {
            lines: self
                .lines
                .into_iter()
                .zip(rhs.lines.into_iter())
                .map(|(l, r)| l + r)
                .collect(),
        }
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.lines
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
