# Data of XXZ spin chain simulation

This folder contains the data obtained from a simulation of a 1D antiferromagnetic XXZ spin chain with a external magnetic field of 0.75 strength. 

Each subfolder contains the data for X qubits (`nqub_X`) and 2 encoding and processing layers. For those algorithms with no encoding layers, we considered 2+2 processing layers (same circuit depth).

The label code of the `.txt` files is the following:

- `basic_data_seed_matterlab_lam075.txt`: details of the simulation contained in the folder.
- `variable_counting_seed_matterlab_lam075.txt`: total number of optimized variables for each algorithm.
- `metaVQE_seed_matterlab_lam075.txt`: history of meta-VQE training optimization.
- `metaVQE_train_seed_matterlab_lam075.txt`: results meta-VQE for the training points.
- `metaVQE_test_seed_matterlab_lam075.txt`: results meta-VQE for the test points.
- `metaVQE_no_encoding_seed_matterlab_lam075.txt`: history of GA-VQE training optimization.
- `metaVQE_no_encoding_test_seed_matterlab_lam075.txt`: results GA-VQE for the test points.
- `standard_VQE_seed_matterlab_lam075.txt`: history of VQE optimization (one for each training point).
- `standard_VQE_test_seed_matterlab_lam075.txt`: results VQE for the training points.
- `standard_VQE_init_metaVQE_seed_matterlab_lam075.txt`: history of opt-meta-VQE optimization.
- `standard_VQE_test_init_metaVQE_seed_matterlab_lam075.txt`: results opt-meta-VQE for the test points.
- `standard_VQE_init_metaVQE_noenc_seed_matterlab_lam075.txt`: history of opt-GA-VQE optimization.
- `standard_VQE_test_init_metaVQE_noenc_seed_matterlab_lam075.txt`: results opt-GA-VQE for the test points.


