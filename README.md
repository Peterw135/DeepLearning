## Autoencoders

This branch features the autoencoder architectures that we attempted.

`autoencoders.py` contains several autoencoder architectures (their nn.Module definitions), and `autoencoder_demonstration.ipynb` contains a demonstration of two of these models. It is a Jupyter notebook that is compatible with Colab. If you wish to run it locally, python3.12 with any pytorch install is likely to be compatible.

`test.txt` and `train.txt` contain the LeafSnap splits, but other files are required for these architectures. `misctrain*.txt` map the other LeafSnap images. You will also need to download the Flavia dataset from https://flavia.sourceforge.net/.

`best_checkpoints/` contains some checkpoints for the most performant model.