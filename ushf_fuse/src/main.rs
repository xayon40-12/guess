use regex::Regex;
fn folders(name: &String) -> Vec<String> {
    let re = Regex::new(&format!(r"^{}\d+$",name)).unwrap();
    std::fs::read_dir(".").expect("Could not enumerate folders in current dir.")
        .filter_map(|d| d.ok().and_then(|e| 
                if let Some(true) = e.file_type().ok().and_then(|t| Some(t.is_dir())) {
                    e.file_name().into_string().ok()
                } else { None }))
        .filter(|f| re.is_match(&f))
        .collect::<Vec<_>>()
}

fn files(name: &String) -> Vec<Vec<File>> {
    folders(name).map(|f| f.
    for f in folders {
        println!("{}", f);
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        panic!("There must be one argument to ushf_fuse which is the name of the simulations to fuse (without the number.")
    } else {
        let name = &args[1];
        let targetstr = format!("{}_fuse",name);
        let target = std::path::Path::new(&targetstr);
        std::fs::create_dir_all(&target).expect(&format!("Could not create destination directory \"{:?}\"", &target));
        let files = files(name);
    }
}
