pub struct ConcurrentHDF5 {
    name: String,
    lock: fslock::LockFile,
}

#[derive(Debug)]
pub enum ConcurrentHDF5Error {
    HDF5Error(hdf5::Error),
    FslockError(fslock::Error),
}

use hdf5::{H5Type, Ix};
use ndarray::{ArrayView, Dimension};
use ConcurrentHDF5Error::*;

pub struct HDF5Data<T> {
    pub data: Vec<T>,
    pub shape: Vec<Ix>,
}

pub enum AttributeType {
    Group,
    DataSet,
}

impl ConcurrentHDF5 {
    pub fn new(name: &str) -> Result<ConcurrentHDF5, ConcurrentHDF5Error> {
        let (path, name) = if let Some(p) = name.rfind('/') {
            (&name[..p + 1], &name[p + 1..])
        } else {
            ("", name)
        };
        let lock =
            fslock::LockFile::open(&format!("{}.{}.lock", path, name)).map_err(FslockError)?; // TODO: use full path to file as a nome where the '/' were replaced by '#' and store the lock file in /tmp
        Ok(ConcurrentHDF5 {
            name: name.into(),
            lock,
        })
    }
    pub fn read_data<T: H5Type>(&mut self, path: &str) -> Result<HDF5Data<T>, ConcurrentHDF5Error> {
        self.lock.lock().map_err(FslockError)?;
        let file = hdf5::File::append(&self.name);
        let data = file
            .and_then(|f| {
                f.dataset(path)
                    .and_then(|d| {
                        d.read_raw().and_then(|data| {
                            Ok(HDF5Data {
                                data,
                                shape: d.shape(),
                            })
                        })
                    })
                    .and_then(|d| f.close().map(|()| d))
            })
            .map_err(HDF5Error);
        self.lock.unlock().map_err(FslockError)?;
        data
    }

    fn prepare_group(
        &mut self,
        file: &mut hdf5::File,
        path: &str,
    ) -> Result<hdf5::Group, hdf5::Error> {
        let paths = path.split("/").collect::<Vec<_>>();
        let mut group = file.group("/");
        for p in &paths {
            group = group.and_then(|g| {
                g.member_names().and_then(|m| {
                    if m.contains(&p.to_string()) {
                        g.group(p)
                    } else {
                        g.create_group(p)
                    }
                })
            });
        }
        group
    }

    pub fn write_data<'d, A, T, D>(
        &mut self,
        path: &str,
        name: &str,
        data: A,
    ) -> Result<(), ConcurrentHDF5Error>
    where
        A: Into<ArrayView<'d, T, D>>,
        T: H5Type,
        D: Dimension,
    {
        self.lock.lock().map_err(FslockError)?;
        let err = hdf5::File::append(&self.name)
            .and_then(|mut f| {
                self.prepare_group(&mut f, path)
                    .and_then(|g| g.new_dataset_builder().with_data(data).create(name))
                    .and_then(|_| f.close())
            })
            .map_err(HDF5Error);
        self.lock.unlock().map_err(FslockError)?;
        err
    }

    pub fn update_attr<T: H5Type>(
        &mut self,
        path: &str,
        attr_name: &str,
        attr_type: AttributeType,
        fun: impl Fn(T) -> T,
        default: &T,
    ) -> Result<T, ConcurrentHDF5Error> {
        self.lock.lock().map_err(FslockError)?;
        let err = hdf5::File::append(&self.name)
            .and_then(|mut f| {
                let loc: hdf5::Result<hdf5::Location> = match attr_type {
                    AttributeType::Group => self
                        .prepare_group(&mut f, path)
                        .and_then(|g| g.as_location()),
                    AttributeType::DataSet => f.dataset(path).and_then(|l| l.as_location()),
                };
                loc.and_then(|g| {
                    let prep = g.attr_names().and_then(|n| {
                        if n.contains(&attr_name.to_string()) {
                            Ok(())
                        } else {
                            g.new_attr::<T>()
                                .create(attr_name)
                                .and_then(|g| g.write_scalar(default))
                        }
                    });
                    let updated = g.attr(attr_name).and_then(|attr| {
                        attr.read_scalar::<T>().and_then(|d| {
                            let updated = fun(d);
                            attr.write_scalar(&updated).and(Ok(updated))
                        })
                    });
                    prep.and(updated)
                })
            })
            .map_err(HDF5Error);
        self.lock.unlock().map_err(FslockError)?;

        err
    }
}
