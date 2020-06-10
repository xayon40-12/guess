use regex::Regex;
use std::io::BufReader;
use std::fs::File;

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
    dst_observable: Vec<BufReader<File>>,
}

impl Data {
    pub fn new(param: String, src_observable: Vec<Vec<BufReader<File>>>, dst_observable: Vec<BufReader<File>>) -> Data {
        Data{ param, src_observable, dst_observable }
    }

    pub fn doit(self) {

    }
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

    let targetstr = format!("{}_fuse",name);
    let target = std::path::Path::new(&targetstr);
    std::fs::create_dir_all(&target).expect(&format!("Could not create destination directory \"{:?}\"", &target));
    let dst_observable = names.into_iter()
        .map(|n| BufReader::new(File::create(&format!("{}/{}", &targetstr, n)).expect("Could not create destination file observable.")))
        .collect::<Vec<_>>();

    Data::new(param, src_observable, dst_observable)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        panic!("There must be one argument to ushf_fuse which is the name of the simulations to fuse (without the number.")
    } else {
        let name = &args[1];
        let data = open_files(name);
        println!("{:#?}", data);
    }
}
