# CPPIF: A multi-objective comprehensive framework for predicting protein-peptide interactions and binding residues

In this work, we developed a multi-objective comprehensive framework, called CPPIF, to predict both binary protein-peptide interaction and their binding residues. We also constructed a benchmark dataset containing more than 8,900 protein-peptide interacting pairs with non-covalent interactions and their corresponding binding residues to systematically evaluate the performances of existing models. Comprehensive evaluation on the benchmark datasets demonstrated that CPPIF can successfully predict the non-covalent protein-peptide interactions that cannot be effectively captured by previous prediction methods. Moreover, CPPIF outperformed other state-of-the-art methods in predicting binding residues in the peptides and achieved good performance in the identification of important binding residues in the proteins.

## The architecture of the proposed CPPIF

<img width="800" alt="image" src="https://github.com/user-attachments/assets/73cc748d-eca7-48e7-aadc-d97a2a326123" />

The overall network architecture of CPPIF has three main modules shown in the Figure: (i) Feature embedding module, (ii) Symmetric encoder module, and (iii) Classification module.
In module (i), we first obtain many residue-level features involved with amino acid type, secondary structure, physicochemical properties, intrinsic disorder tendency, and evolutionary information from the protein and peptide sequences. To avoid the inconsistency of different types of features, we further divide them into regression features (e.g., intrinsic disorder scores and position-specific scoring matrix (PSSM)) and classified features (e.g., amino acid type, secondary structure, polarity, and hydrophilicity) and apply two different embedding strategies to them separately. After that, all the above feature embeddings of every amino acid are concatenated into a pre-processed embedding vector; thus, the protein or peptide sequence is converted to an embedding matrix. 

In module (ii), CPPIF exploits a symmetric encoder architecture to extract the features of protein and peptide in the input pair. Specifically, the symmetric encoder contains three blocks: the protein text CNN block, peptide text CNN block, and BERT encoder block. For each text CNN block, the embedding matrix from the module (i) is input into it to extract local residue contextual features. Moreover, to better capture the relationship between the protein and peptide in the input pairs and the long-distance dependence of sequences, we connect the protein and peptide sequences of one pair first and encode them together with BERT encoder block. As a result, the output of the BERT encoder block is the feature representation after mutual attention of any residue in the protein-peptide pair, which is capable of reflecting the contribution of each residue for the protein-peptide interaction and then split into representations of protein and peptide. 

In module (iii), CPPIF adopts three different feature combinations for three tasks. For the binary interaction prediction, all four features are passing through the max pooling layer, concatenated together, and fed into three fully connected layers to determine whether the protein-peptide pair is interacting or not. To identify the binding residues on proteins, our model combines the output of the protein text CNN block and the protein representation from the BERT encoder block and puts it into a fully connected neural network with three layers to predict whether each residue is binding or not. The prediction of binding residues on peptides is similar to proteins but uses the combined features of the output from the peptide text CNN block and the peptide representation from the BERT encoder block.


## How to use it

The main program in the train folder main_cppif.py file.  File main_cppif.py has detail notes. You could change configuration/config_CPPIF.py to achieve custom training and testing, such as modifying datasets, setting hyperparameters and so on. For example if you want to change the padding sequence length you can change the following raws in config.CPPIF.py.

```python
parse.add_argument('-pad-pep-len', type=int, default=50, help='number of sense in multi-sense')
parse.add_argument('-pad-prot-len', type=int, default=679, help='number of sense in multi-sense')
```

In the model folder you can find our model CPPIF as well as our comparied models. And in the folder util it has data_loader_cppif.py which includes how to process and load the data into model.


The project is mainly implemented through **Pytorch**, **Numpy** and **transformers**.

## Contact

For further questions or details, reach out to Ruheng Wang (wangruheng@mail.sdu.edu.cn)
