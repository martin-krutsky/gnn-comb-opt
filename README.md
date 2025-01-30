# Binarizing PI-GNNs for Combinatorial Optimization

Code repository for experiments related to the paper _Binarizing PI-GNNs for Combinatorial Optimization_, based on [Schuetz et al.](https://www.nature.com/articles/s42256-022-00468-6) and their example [GitHub repository](https://github.com/amazon-science/co-with-gnns-example).

## Running the Experiments
Install the necessary libraries via `pip install -r requirements.txt`

The training entry point is located in `scripts/run.py`, for available command line arguments, see `scripts/parser.py`.

## Implementation of Discretization/Fuzzy Logic
The discrete operations are implemented in `utils/discretize.py`.
The fuzzy operations are taken (with minimal modifications) from the [PyTorch implementation of Logic Tensor Networks](https://github.com/tommasocarraro/LTNtorch).
