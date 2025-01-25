# CPPIF: A multi-objective comprehensive framework for predicting protein-peptide interactions and binding residues

In this work, we developed a multi-objective comprehensive framework, called CPPIF, to predict both binary protein-peptide interaction and their binding residues. We also constructed a benchmark dataset containing more than 8,900 protein-peptide interacting pairs with non-covalent interactions and their corresponding binding residues to systematically evaluate the performances of existing models. Comprehensive evaluation on the benchmark datasets demonstrated that CPPIF can successfully predict the non-covalent protein-peptide interactions that cannot be effectively captured by previous prediction methods. Moreover, CPPIF outperformed other state-of-the-art methods in predicting binding residues in the peptides and achieved good performance in the identification of important binding residues in the proteins.


## How to use it

The main program in the train folder main_cppif.py file.  File main_cppif.py has detail notes. You could change configuration/config_CPPIF.py to achieve custom training and testing, such as modifying datasets, setting hyperparameters and so on. For example if you want to change the padding seqence length you can change the following raws in config.CPPIF.py.

```python
parse.add_argument('-pad-pep-len', type=int, default=50, help='number of sense in multi-sense')
parse.add_argument('-pad-prot-len', type=int, default=679, help='number of sense in multi-sense')
```

In the model folder you can find our model CPPIF as well as our comparied models. And in the folder util it has data_loader_cppif.py which includes how to process and load the data into model.


The project is mainly implemented through **Pytorch**, **Numpy** and **transformers**.

## Contact

For further questions or details, reach out to Ruheng Wang (wangruheng@mail.sdu.edu.cn)
