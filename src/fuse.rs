use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Write;
use std::ops::Add;

fn check(name: &str, val: &str) -> bool {
    let mut i = 0;
    let name = name.chars().collect::<Vec<_>>();
    let val = val.chars().collect::<Vec<_>>();
    let l = name.len();
    while i < l && name[i] == val[i] {
        i += 1;
    }
    let l = val.len();
    while i < l && val[i].is_digit(10) {
        i += 1;
    }
    i == l
}
fn folders(name: &str) -> Vec<String> {
    std::fs::read_dir(".")
        .expect("Could not enumerate folders in current dir.")
        .filter_map(|d| {
            d.ok().and_then(|e| {
                if let Some(true) = e.file_type().ok().and_then(|t| Some(t.is_dir())) {
                    e.file_name().into_string().ok()
                } else {
                    None
                }
            })
        })
        .filter(|f| check(name, f))
        .collect::<Vec<_>>()
}

fn files(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir)
        .expect(&format!("Could not enumerate folders in dir \"{}\".", dir))
        .map(|d| {
            d.expect("Error reading direcory.")
                .file_name()
                .into_string()
                .expect("Error converting file name.")
        })
        .filter(|d| d != "config")
        .map(|d| format!("{}/{}", dir, d))
        .collect::<Vec<_>>()
}

#[derive(Debug)]
struct Data {
    param: String,
    src_observable: Vec<Vec<BufReader<File>>>,
    dst_observable: Vec<(String, File)>,
    fuse_file: File,
}

type ValueT = Vec<f64>;

#[derive(Debug, Clone)]
enum ArrayT {
    Values(ValueT),
    WithCoord(f64, ValueT),
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

impl ArrayT {
    fn to_string(&self) -> String {
        match self {
            ArrayT::Values(v) => v
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            ArrayT::WithCoord(c, v) => format!(
                "{};{}",
                c.to_string(),
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        }
    }
    fn power(self, n: usize) -> ArrayT {
        match self {
            ArrayT::Values(v) => {
                ArrayT::Values(v.into_iter().map(|i| f64::powf(i, n as f64)).collect())
            }
            ArrayT::WithCoord(c, v) => {
                ArrayT::WithCoord(c, v.into_iter().map(|i| f64::powf(i, n as f64)).collect())
            }
        }
    }
    fn divide(self, n: usize) -> ArrayT {
        match self {
            ArrayT::Values(v) => ArrayT::Values(v.into_iter().map(|i| i / n as f64).collect()),
            ArrayT::WithCoord(c, v) => {
                ArrayT::WithCoord(c, v.into_iter().map(|i| i / n as f64).collect())
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Line {
    time: f64,
    field: String,
    observable: String,
    array: Vec<ArrayT>,
}

#[derive(Debug, Clone)]
struct FuseLine {
    time: f64,
    field: String,
    observable: String,
    array: (usize, Vec<Vec<ArrayT>>),
}

impl From<String> for Line {
    fn from(source: String) -> Self {
        let vals = source.trim().split("|").collect::<Vec<_>>();
        if vals.len() != 4 {
            panic!("A data line should be of the form:\ntime|field|observable|pos;c1,c2 pos;c1,c2 ...\nWhere pos is a number that might no be present (then the ';' would not be present either) and c1 c2 ... are the values (there might be many separated by comas, for instance a complex number would have two).")
        }
        Line {
            time: vals[0].parse::<f64>().expect(&format!(
                "Time cannot be parsed while reading line:\n{}",
                source
            )),
            field: vals[1].to_string(),
            observable: vals[2].to_string(),
            array: vals[3]
                .split(" ")
                .map(|v| {
                    let tmp = v.split(";").collect::<Vec<_>>();
                    match tmp.len() {
                        1 => ArrayT::Values(
                            tmp[0]
                                .split(",")
                                .map(|i| {
                                    i.parse::<f64>().expect(&format!(
                                        "Cannot parse \"{}\" as a number in: \"{}\"",
                                        i, v
                                    ))
                                })
                                .collect(),
                        ),
                        2 => ArrayT::WithCoord(
                            tmp[0].parse::<f64>().expect(&format!(
                                "Cannot parse \"{}\" as a number in: \"{}\"",
                                tmp[0], v
                            )),
                            tmp[1]
                                .split(",")
                                .map(|i| {
                                    i.parse::<f64>().expect(&format!(
                                        "Cannot parse \"{}\" as a number in: \"{}\"",
                                        i, v
                                    ))
                                })
                                .collect(),
                        ),
                        _ => panic!(
                            "There should be at most one ';' when parsing values: \"{}\"",
                            v
                        ),
                    }
                })
                .collect(),
        }
    }
}

impl FuseLine {
    fn to_fuse(source: Line, n: usize) -> Self {
        let array = source.array;
        Self {
            time: source.time,
            field: source.field,
            observable: source.observable,
            array: (
                1,
                (0..n)
                    .map(|i| array.clone().into_iter().map(|a| a.power(i + 1)).collect())
                    .collect(),
            ),
        }
    }
    fn similar(&self, other: &FuseLine) -> bool {
        self.time == other.time
            && self.field == other.field
            && self.observable == other.observable
            && self.array.1.len() == other.array.1.len()
    }
    fn to_string(&self) -> String {
        let n = self.array.0;
        self.array
            .1
            .iter()
            .enumerate()
            .map(|(i, a)| {
                format!(
                    "{}|{}|{}|{}",
                    self.time.to_string(),
                    self.field,
                    format!("<({})^{}>", self.observable, i + 1),
                    a.clone()
                        .into_iter()
                        .map(|v| v.divide(n).to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }
}

impl Add<FuseLine> for FuseLine {
    type Output = FuseLine;
    fn add(self, rhs: FuseLine) -> Self::Output {
        if !self.similar(&rhs) {
            panic!("Only similar FuseLine can be added.")
        }
        FuseLine {
            time: self.time,
            field: self.field,
            observable: self.observable,
            array: (
                self.array.0 + rhs.array.0,
                self.array
                    .1
                    .into_iter()
                    .zip(rhs.array.1.into_iter())
                    .map(|(c1, c2)| {
                        c1.into_iter()
                            .zip(c2.into_iter())
                            .map(|(a1, a2)| a1 + a2)
                            .collect()
                    })
                    .collect(),
            ),
        }
    }
}

impl Data {
    pub fn new(
        param: String,
        src_observable: Vec<Vec<BufReader<File>>>,
        dst_observable: Vec<(String, File)>,
        fuse_file: File,
    ) -> Data {
        Data {
            param,
            src_observable,
            dst_observable,
            fuse_file,
        }
    }

    pub fn doit(mut self) {
        let fuse = self
            .dst_observable
            .iter()
            .zip(self.src_observable.iter())
            .map(|((name, _), src)| format!("  \"{}\": {},", name, src.len()))
            .collect::<Vec<_>>()
            .join("\n");
        write!(self.fuse_file, "{{\n{}}}", fuse).unwrap();
        self.dst_observable
            .into_par_iter()
            .zip(self.src_observable.into_par_iter())
            .for_each(|((dstname, mut dst), mut src)| {
                let mut id = 0;
                let n = 4;
                'top: loop {
                    id += 1;
                    let mut acc: Option<FuseLine> = None;
                    for s in &mut src {
                        let mut line = String::new();
                        s.read_line(&mut line).unwrap();
                        if line.len() == 0 {
                            break 'top;
                        }
                        let l = FuseLine::to_fuse(Line::from(line), n);
                        if let Some(acc1) = acc {
                            if !acc1.similar(&l) {
                                panic!(
                                    "Lines {} does not correspond in observable \"{}\".",
                                    id, &dstname
                                );
                            }
                            acc = Some(acc1 + l);
                        } else {
                            acc = Some(l);
                        }
                    }

                    write_array(&mut dst, acc.unwrap());
                }
            });
    }
}

fn write_array(dst: &mut File, line: FuseLine) {
    write!(dst, "{}", line.to_string()).unwrap();
}

fn open_files(name: &str) -> Data {
    let folders = folders(name);
    let mut param = None;
    let mut names = None;
    let src_observable = folders
        .iter()
        .map(|f| {
            let tmpparam = std::fs::read_to_string(&format!("{}/config/param.ron", f))
                .expect("Could not read config/param.ron");
            if let Some(param) = &param {
                if param != &tmpparam {
                    panic!("The content of the param.ron files are different, abort!");
                }
            } else {
                param = Some(tmpparam);
            }
            let mut tmpnames = vec![];
            let observable = files(f)
                .into_iter()
                .map(|f| {
                    tmpnames.push(f[f.rfind("/").unwrap() + 1..].to_string());
                    BufReader::new(File::open(&f).expect(&format!("Could not open file \"{}\"", f)))
                })
                .collect::<Vec<_>>();
            if let Some(names) = &names {
                if names != &tmpnames {
                    panic!("The obsarvable have not the same names in each folder, abort!");
                }
            } else {
                names = Some(tmpnames);
            }

            observable
        })
        .collect::<Vec<_>>();
    let param = param.expect("No param.ron found.");
    let names = names.expect("No observable found.");
    let mut itr = src_observable.into_iter();
    let mut src_observable = itr
        .next()
        .unwrap()
        .into_iter()
        .map(|i| vec![i])
        .collect::<Vec<_>>();
    for it in itr {
        it.into_iter()
            .enumerate()
            .for_each(|(i, v)| src_observable[i].push(v));
    }

    let targetstr = format!("{}_fuse", name);
    let configstr = format!("{}/config", targetstr);
    let target = std::path::Path::new(&configstr);
    std::fs::create_dir_all(&target).expect(&format!(
        "Could not create destination directory \"{:?}\"",
        &target
    ));
    let dst_observable = names
        .into_iter()
        .map(|n| {
            (
                n.clone(),
                File::create(&format!("{}/{}", &targetstr, n))
                    .expect("Could not create destination file observable."),
            )
        })
        .collect::<Vec<_>>();

    let fuse_file = File::create(&format!("{}/fuse.ron", &configstr))
        .expect("Could not create destination file fuse.");
    let mut param_file = File::create(&format!("{}/param.ron", &configstr))
        .expect("Could not create destination file param.");
    write!(param_file, "{}", param).unwrap();

    Data::new(param, src_observable, dst_observable, fuse_file)
}

pub fn fuse(args: Vec<String>) {
    if args.len() != 1 {
        panic!("There must be one argument to guess_fuse which is the name of the simulations to fuse (without the number.")
    } else {
        let name = &args[0];
        let data = open_files(name);
        data.doit();
    }
}
