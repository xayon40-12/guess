# General User-oriented Effortless Stochastic Solver

## Install

To build you will need `rustc` with `cargo` installed (the `rustup` installer is a simple way to obtain both).
Then clone the project and build it in release mode:  
```sh
git clone https://github.com/xayon40-12/guess
cd guess
cargo build --release
```
Now in order to use `guess`, simply add the path provided by the following command to your `PATH` environment variable:  
```sh
echo $(pwd)/utils
```

## Usage

In odrer to run a simulation. First create an parameter file (for instance `param.ron`) and then use `guess` to run it:  
```sh
guess run param.ron
```
If you run simulations that involve noise and want multiple of them, simply add the quantity you want after the parameter file name separated by a colon:  
```sh
guess run param.ron:10
```


## Parameter file

TODO

See examples in the `param/` folder.
