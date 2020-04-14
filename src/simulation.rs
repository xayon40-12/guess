use gpgpu::Handler;
use gpgpu::data_file::Format;
use crate::{Param,Callback};


pub struct Simulation {
    handler: Handler,
    callbacks: Vec<Callback>,
}

impl Simulation {
    pub fn from_param<'a>(file_name: &'a str) -> gpgpu::Result<Self> {
        let param: Param = serde_yaml::from_str(&std::fs::read_to_string(file_name).expect(&format!("Could not find parameter file \"{}\".", file_name))).unwrap();
        println!("param:\n{:?}", &param);
        let mut handler = Handler::builder()?;
        for f in &param.data_files {
            handler = handler.load_data(f,Format::Column(&std::fs::read_to_string(format!("{}.txt",f)).expect(&format!("Could not find data file \"{}\".", file_name))),false,None); //TODO autodetect format from file extension
        }

        let dt = 1.0; //TODO compute dt from param symbols

        let callbacks = param.actions.iter().map(|a| a.to_callback(dt)).collect();

        let handler = handler.build()?;

        Ok(Simulation { handler, callbacks })
    }
}
