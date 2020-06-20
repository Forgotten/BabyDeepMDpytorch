# Baby Deep MD (pytorch version)

This is a light weight version of Deep MD written in pytorch, which mainly serves as a test bench for developping prototypes. 

## Requirements:

- Pytorch > 1.4
- Numba
- Numpy 

We have included some data and examples of the configuration files.
The scripts are going to generate the data on the fly, save it, so it can be used afterwards

The easiest way to run the examples, is to go to the run/Np20 folder (containing runs with 20 particles)
and then run 

python ../../src/2BodyEnergyDist.py Np20_Per_mu_10_short.json

The test files are supposed to be run on their own. They are mostly intendend to show and test the inner strucutre of the Models in src/networks.py 