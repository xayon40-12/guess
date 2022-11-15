pub struct ConcurrentHDF5 {
    file: hdf5::File,
    lock: fslock::LockFile,
}

#[derive(Debug)]
pub enum ConcurrentHDF5Error {
    HDF5Error(hdf5::Error),
    FslockError(fslock::Error),
}

use hdf5::H5Type;
use ConcurrentHDF5Error::*;

impl ConcurrentHDF5 {
    pub fn new() -> Result<ConcurrentHDF5, ConcurrentHDF5Error> {
        let file = hdf5::File::append("data.h5").map_err(HDF5Error)?;
        let lock = fslock::LockFile::open("data.lock").map_err(FslockError)?;
        Ok(ConcurrentHDF5 { file, lock })
    }
    pub fn write_data(&mut self, path: &str, data: &[f64]) -> Result<(), ConcurrentHDF5Error> {
        self.lock.lock().map_err(FslockError)?;
        let paths = path.split("/").collect::<Vec<_>>();
        let name = paths[paths.len() - 1];
        let mut group = self.file.group("/").map_err(HDF5Error)?;
        for p in &paths[0..paths.len() - 1] {
            if group
                .member_names()
                .map_err(HDF5Error)?
                .contains(&p.to_string())
            {
                group = group.group(p).map_err(HDF5Error)?;
            } else {
                group = group.create_group(p).map_err(HDF5Error)?;
            }
        }
        group
            .new_dataset_builder()
            .with_data(data)
            .create(name)
            .map_err(HDF5Error)?;
        self.lock.unlock().map_err(FslockError)
    }

    pub fn update_group_attr<T: H5Type>(
        &mut self,
        path: &str,
        attr_name: &str,
        f: impl Fn(T) -> T,
        default: &T,
    ) -> Result<T, ConcurrentHDF5Error> {
        self.lock.lock().map_err(FslockError)?;
        let group = self.file.group(path).map_err(HDF5Error)?;
        if !group
            .attr_names()
            .map_err(HDF5Error)?
            .contains(&attr_name.to_string())
        {
            group
                .new_attr::<T>()
                .create(attr_name)
                .map_err(HDF5Error)?
                .write_scalar(default)
                .map_err(HDF5Error)?;
        };
        let attr = group.attr(attr_name).map_err(HDF5Error)?;
        let data = attr.read_scalar::<T>().map_err(HDF5Error)?;
        let updated = f(data);
        attr.write_scalar(&updated).map_err(HDF5Error)?;

        self.lock.unlock().map_err(FslockError)?;

        Ok(updated)
    }
}
