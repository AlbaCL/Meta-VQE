# Meta-VQE

This is the repository for the article "The Meta-Variational Quantum Eigensolver (Meta-VQE): Learning energy profiles of parameterized Hamiltonians for quantum simulation", Alba Cervera-Lierta, Jakob S. Kottmann, Alán Aspuru-Guzik, [arXiv:2009.13545[quant-ph]](https://arxiv.org/abs/2009.13545).

__Content:__

* `data` folder: contains all data presented in the main article.
* `img` folder: contains the plots of the main article and other plots generated with the available data.
* `Meta-VQE` demo: jupyter notebook with the source code used to run the simulations. 

__Dependencies:__

You will need `tequila` to run the notebooks.  
Just follow the instructions on the [github page](https://github.com/aspuru-guzik-group/tequila).
If you are using Linux, you can clone this repository and run `pip install .` to install `tequila` form the `setup.py` file provided here.  
If you are using Mac or Windows, you probably need to install the `qulacs` simulator manually (or alternatively any other suported quantum backend) -- see the `tequila` [github page](https://github.com/aspuru-guzik-group/tequila). 
