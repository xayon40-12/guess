use regex::Regex;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::io::Write;
use rayon::prelude::*;

fn folders(name: &str) -> Vec<String> {
    let re = Regex::new(&format!(r"^{}\d+$",name)).unwrap();
    std::fs::read_dir(".").expect("Could not enumerate folders in current dir.")
        .filter_map(|d| d.ok().and_then(|e| 
                if let Some(true) = e.file_type().ok().and_then(|t| Some(t.is_dir())) {
                    e.file_name().into_string().ok()
                } else { None }))
        .filter(|f| re.is_match(&f))
        .collect::<Vec<_>>()
}

fn files(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir).expect(&format!("Could not enumerate folders in dir \"{}\".", dir))
        .map(|d| d.expect("Error reading direcory.")
                  .file_name().into_string().expect("Error converting file name.")
        ).filter(|d| d != "config")
        .map(|d| format!("{}/{}", dir, d))
        .collect::<Vec<_>>()
}

#[derive(Debug)]
struct Data {
    param: String,
    src_observable: Vec<Vec<BufReader<File>>>,
    dst_observable: Vec<(String,File)>,
    fuse_file: File,
}

impl Data {
    pub fn new(param: String, src_observable: Vec<Vec<BufReader<File>>>, dst_observable: Vec<(String,File)>, fuse_file: File) -> Data {
        Data{ param, src_observable, dst_observable, fuse_file }
    }

    pub fn doit(mut self) {
        let fuse = self.dst_observable.iter().zip(self.src_observable.iter()).map(|((name,_),src)| format!("{}: {}", name, src.len())).collect::<Vec<_>>().join("\n");
        write!(self.fuse_file, "{}", fuse).unwrap();
        self.dst_observable.into_par_iter().zip(self.src_observable.into_par_iter()).for_each(|((dstname,mut dst),mut src)| {
            let mut id = 0;
            let mut last: Option<(String,Vec<f64>,Vec<f64>)> = None;
            let mut line = String::new();
            let num = src.len();
            let mut is_sigma;
            'top: loop {
                id += 1;
                let mut mean: Option<(Vec<f64>,Vec<f64>)> = None;
                let mut start: Option<String> = None;
                is_sigma = false;
                for s in &mut src {
                    line = String::new();
                    s.read_line(&mut line).unwrap();
                    if line.len() == 0 { break 'top; }
                    let l = line.split(":").collect::<Vec<_>>();
                    if let Some(start) = &start {
                        if start != &l[0] { panic!("Lines {} does not correspond in observable \"{}\".", id, &dstname); }
                    } else {
                        start = Some(l[0].to_string());
                        is_sigma = l[0][l[0].rfind(" ").unwrap()+1..].starts_with("sigma_");
                    }
                    if l[1].starts_with(" [") {
                        let vec = l[1][2..l[1].rfind("]").unwrap()]
                            .split(",")
                            .map(|i| i.parse::<f64>().unwrap())
                            .collect::<Vec<_>>();
                        let sig = vec.iter().map(|i| i*i).collect::<Vec<_>>();
                        if is_sigma {
                            if let Some((mean,_)) = &mut mean {
                                let m = last.clone().expect(&format!("There must be an observable befor its sigma_ line {} in \"{}\".", id, &dstname)).2;
                                for i in 0..vec.len() {
                                    mean[i] += sig[i]+m[i]*m[i];
                                }
                            } else {
                                mean = Some((sig,vec![]));
                            }
                        } else {
                            if let Some((mean,sigma)) = &mut mean {
                                for i in 0..vec.len() {
                                    mean[i] += vec[i];
                                    sigma[i] += sig[i];
                                }
                            } else {
                                mean = Some((vec,sig));
                            }
                        }
                    }
                }
                let start = start.unwrap();
                if is_sigma {
                    if let Some((mut sig,_)) = mean {
                        let m = last.unwrap().2;
                        for i in 0..sig.len() { sig[i] = (sig[i]/num as f64-m[i]*m[i]).sqrt(); }
                        write_array(&mut dst,&start,&sig);
                    }
                    last = None;
                } else {
                    if let Some(last) = &last {
                        write_array(&mut dst,&last.0,&last.1);
                    }
                    last = None;
                    if let Some((mut mean,mut sigma)) = mean {
                        let pos = start.rfind(" ").unwrap();
                        let sigstart = format!("{}sigma_{}", &start[0..pos+1], &start[pos+1..]);
                        for i in 0..mean.len() { mean[i] /= num as f64; }
                        for i in 0..sigma.len() { sigma[i] = (sigma[i]/num as f64-mean[i]*mean[i]).sqrt(); }
                        write_array(&mut dst,&start,&mean);
                        last = Some((sigstart, sigma, mean));
                    } else {
                        write!(dst, "{}", line).unwrap();
                    }
                }

            }

            if !is_sigma {
                if let Some(last) = &last {
                    write_array(&mut dst,&last.0,&last.1);
                }
            }
        });
    }
}

fn write_array(dst: &mut File, start: &str, arr: &Vec<f64>) {
    write!(dst, "{}: [{}]\n", start, arr.iter().map(|i| format!("{:e}",i)).collect::<Vec<_>>().join(",")).unwrap();
}

fn open_files(name: &str) -> Data {
    let folders = folders(name);
    let mut param = None;
    let mut names = None;
    let src_observable = folders.iter().map(|f| {
        let tmpparam = std::fs::read_to_string(&format!("{}/config/param.yaml",f))
            .expect("Could not read config.yaml");
        if let Some(param) = &param {
            if param != &tmpparam {
                panic!("The content of the param.yaml files are different, abort!");
            }
        } else {
            param = Some(tmpparam);
        }
        let mut tmpnames = vec![];
        let observable = files(f)
            .into_iter()
            .map(|f| {
                tmpnames.push(f[f.rfind("/").unwrap()+1..].to_string());
                BufReader::new(File::open(&f).expect(&format!("Could not open file \"{}\"",f)))
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
    }).collect::<Vec<_>>();
    let param = param.expect("No param.yaml found.");
    let names = names.expect("No observable found.");
    let mut itr = src_observable.into_iter();
    let mut src_observable = itr.next().unwrap().into_iter().map(|i| vec![i]).collect::<Vec<_>>();
    for it in itr {
        it.into_iter().enumerate().for_each(|(i,v)| src_observable[i].push(v));
    }

    let targetstr = format!("{}_fuse",name);
    let configstr = format!("{}/config",targetstr);
    let target = std::path::Path::new(&configstr);
    std::fs::create_dir_all(&target).expect(&format!("Could not create destination directory \"{:?}\"", &target));
    let dst_observable = names.into_iter()
        .map(|n| (n.clone(),File::create(&format!("{}/{}", &targetstr, n)).expect("Could not create destination file observable.")))
        .collect::<Vec<_>>();

    let fuse_file = File::create(&format!("{}/fuse.yaml", &configstr)).expect("Could not create destination file fuse.");
    let mut param_file = File::create(&format!("{}/param.yaml", &configstr)).expect("Could not create destination file param.");
    write!(param_file, "{}", param).unwrap();

    Data::new(param, src_observable, dst_observable, fuse_file)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        panic!("There must be one argument to ushf_fuse which is the name of the simulations to fuse (without the number.")
    } else {
        let name = &args[1];
        let data = open_files(name);
        data.doit();
    }
}
