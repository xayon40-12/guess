use nom::branch::alt;
use nom::bytes::complete::take_until1;
use nom::character::complete::char;
use nom::character::complete::u64;
use nom::combinator::eof;
use nom::combinator::map;
use nom::combinator::opt;
use nom::multi::separated_list1;
use std::fmt;
use std::ops::Add;
use std::ops::DivAssign;

use nom::number::complete::double;
use nom::sequence::*;
use nom::IResult;

use super::array_t::ArrayT;

#[derive(Debug, Clone)]
pub struct Line {
    // time|field_name|obs_name|count#vec[0]/vec[1]/...
    time: f64,
    status: Option<String>,
    field_name: String,
    count: usize,          // number of accumulated data(statistics)
    vec: Vec<Vec<ArrayT>>, // vec of vec to store each moments or the vector of data
}

impl Line {
    pub fn similar(&self, other: &Line) -> bool {
        self.time == other.time
            && self.field_name == other.field_name
            && self.status == other.status
            && self.vec.len() == other.vec.len()
            && self.vec[0].len() == other.vec[0].len()
            && self.vec[0][0].len() == other.vec[0][0].len()
    }
    pub fn mean(&mut self) {
        *self /= self.count;
    }
    pub fn is_nan(&self) -> bool {
        self.vec.iter().fold(false, |acc, v| {
            acc || v.iter().fold(false, |acc, v| acc || v.is_nan())
        })
    }
}

impl fmt::Display for Line {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}|{}{}|{}#{}",
            self.time,
            self.status
                .as_ref()
                .map_or("".to_string(), |s| format!("{}|", s)),
            self.field_name,
            self.count,
            self.vec
                .iter()
                .map(|i| i
                    .iter()
                    .map(|j| j.to_string())
                    .collect::<Vec<_>>()
                    .join(" "))
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

impl From<String> for Line {
    fn from(source: String) -> Self {
        (&source[..]).into()
    }
}
impl From<&str> for Line {
    fn from(source: &str) -> Self {
        match parse(source) {
            Ok((_, l)) => l,
            Err(e) => panic!("{}", e),
        }
    }
}

fn nums(s: &str) -> IResult<&str, Vec<f64>> {
    separated_list1(char(';'), double)(s)
}
fn pname(s: &str) -> IResult<&str, String> {
    terminated(take_until1("|"), char('|'))(s).map(|(r, i)| (r, i.to_string()))
}
fn ptime(s: &str) -> IResult<&str, f64> {
    terminated(double, char('|'))(s)
}
fn pcount(s: &str) -> IResult<&str, Option<u64>> {
    opt(terminated(u64, char('#')))(s)
}
fn pvec(s: &str) -> IResult<&str, Vec<Vec<ArrayT>>> {
    let onlynums = |s| nums(s).map(|(r, ns)| (r, ArrayT::Values(ns)));
    let coordnums = |s| {
        tuple((double, char(':'), &nums))(s).map(|(r, (c, _, ns))| (r, ArrayT::WithCoord(c, ns)))
    };
    let data = alt((coordnums, onlynums));
    let vecdata = separated_list1(char(' '), data);
    terminated(separated_list1(char('/'), vecdata), eof)(s)
}
fn parse(input: &str) -> IResult<&str, Line> {
    let (input, (time, status, field_name, count, mut vec)) = alt((
        map(tuple((ptime, pname, pcount, pvec)), |(t, n, c, v)| {
            (t, None, n, c, v)
        }),
        map(
            tuple((ptime, pname, pname, pcount, pvec)),
            |(t, s, n, c, v)| (t, Some(s), n, c, v),
        ),
    ))(input)?;
    let count = count.unwrap_or(1) as usize;
    if count == 1 {
        if vec.len() == 1 {
            // if single simulation with no higher order moments, computer 2nd order
            let num = 4;
            for n in 2..=num {
                let next_ord = vec[0].iter().map(|i| i.clone() ^ n).collect();
                vec.push(next_ord);
            }
        }
    } else {
        vec = vec
            .into_iter()
            .map(|v| v.into_iter().map(|i| i * count).collect()) // multiply by the count to mean properly later
            .collect();
    }
    Ok((
        input,
        Line {
            time,
            status,
            field_name,
            count,
            vec,
        },
    ))
}

impl Add<Line> for Line {
    type Output = Line;
    fn add(self, rhs: Line) -> Self::Output {
        if !self.similar(&rhs) {
            panic!("Only similar Line can be added.")
        }
        Line {
            time: self.time,
            field_name: self.field_name,
            status: self.status,
            count: self.count + rhs.count,
            vec: self
                .vec
                .into_iter()
                .zip(rhs.vec.into_iter())
                .map(|(ls, rs)| {
                    ls.into_iter()
                        .zip(rs.into_iter())
                        .map(|(l, r)| l + r)
                        .collect()
                })
                .collect(),
        }
    }
}

impl DivAssign<usize> for Line {
    fn div_assign(&mut self, n: usize) {
        for i in &mut self.vec {
            for j in i {
                *j /= n;
            }
        }
    }
}
