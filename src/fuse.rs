use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::fs::File;
use std::io::Write;

pub mod array_t;
pub mod data;
pub mod line;
use data::Data;

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
                if let Some(true) = e.file_type().ok().map(|t| t.is_dir()) {
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
        .unwrap_or_else(|_| panic!("Could not enumerate folders in dir \"{}\".", dir))
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
struct Files {
    param: String,
    src_folders: Vec<String>,
    src_names: Vec<String>,
    dst_observable: Vec<(String, File)>,
}

impl Files {
    pub fn new(
        param: String,
        src_folders: Vec<String>,
        src_names: Vec<String>,
        dst_observable: Vec<(String, File)>,
    ) -> Files {
        Files {
            param,
            src_folders,
            src_names,
            dst_observable,
        }
    }

    pub fn doit(self) {
        let mut folders = self.src_folders.clone().into_iter();
        let fst = folders
            .next()
            .unwrap_or_else(|| panic!("There must be at least one folder to fuse."));
        self.dst_observable
            .into_par_iter()
            .zip(self.src_names.into_par_iter())
            .for_each(|((dstname, mut dst), src)| {
                let mut acc = folders
                    .clone()
                    .fold(Data::new(&fst, &src), |acc, i| acc + Data::new(&i, &src));
                acc.finish();
                writeln!(dst, "{}", acc)
                    .unwrap_or_else(|_| panic!("Could not write to {}", dstname));
            });
    }
}

fn search_files(name: &str) -> Files {
    let folders = folders(name);
    let mut param = None;
    let mut names = None;

    folders.iter().for_each(|f| {
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
        files(f).into_iter().for_each(|f| {
            tmpnames.push(f[f.rfind('/').unwrap() + 1..].to_string());
        });
        if let Some(names) = &names {
            if names != &tmpnames {
                panic!("The observables have not the same names in each folder, abort!");
            }
        } else {
            names = Some(tmpnames);
        }
    });

    let param = param.expect("No param.ron found.");
    let names = names.expect("No observable found.");
    let targetstr = format!("{}_fuse", name);
    let configstr = format!("{}/config", targetstr);
    let target = std::path::Path::new(&configstr);
    std::fs::create_dir_all(&target)
        .unwrap_or_else(|_| panic!("Could not create destination directory \"{:?}\"", &target));
    let dst_observable = names
        .iter()
        .map(|n| {
            (
                n.clone(),
                File::create(&format!("{}/{}", &targetstr, n))
                    .expect("Could not create destination file observable."),
            )
        })
        .collect::<Vec<_>>();

    let mut param_file = File::create(&format!("{}/param.ron", &configstr))
        .expect("Could not create destination file param.");
    write!(param_file, "{}", param).unwrap();

    Files::new(param, folders, names, dst_observable)
}

pub fn fuse(args: Vec<String>) {
    if args.len() != 1 {
        panic!("There must be one argument to guess_fuse which is the name of the simulations to fuse (without the number.")
    } else {
        let name = &args[0];
        let files = search_files(name);
        files.doit();
    }
}
