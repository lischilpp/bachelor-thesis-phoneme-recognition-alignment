<div align="center">
  <h1>Bachelor Thesis - <br>Phoneme classification and alignment<br> through recognition on TIMIT</h1>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  </p>
  <p>My bachelor thesis on Phoneme recognition and alignment on the TIMIT dataset</p>
  <p><a href="https://isl.anthropomatik.kit.edu/downloads/BachelorarbeitSchlipp.pdf">Link to publication</a></p>
</div>

## Abstract
In this work we explore a hybrid between ANNs and DTW for phoneme alignment on the TIMIT dataset. The idea is to use the output probabilities of a neural phoneme recognition model together with a probability-based DTW in order to align phonemes.
For phoneme recognition we achieve 18.1% FER which is an 4.0% improvement over the state-of-the-art.
Our alignment results in a 86.3% phoneme boundary accuracy with a 20ms tolerance. Furthermore phoneme classification based on recordings of single phonemes is being tried resulting in an accuracy of 66.68%.
Apart from that we introduce the CyclicPlateauScheduler, a new learning rate scheduler combining triangular cyclic learning rates with ReduceLROnPlateau.

## CNN experiments
The code for the initial CNN experiments can be found [here](https://github.com/lischilpp/bachelor-thesis-phoneme-recognition-alignment_cnn)

## Getting Started
### Dependencies
`pip install seaborn pandas matplotlib torch torchaudio spafe pytorch-lightning torchmetrics dtw-python`
### Configuration
You can adjust several global variables in the [settings.py](https://github.com/lischilpp/bachelor-thesis-phoneme-recognition-alignment/blob/main/src/settings.py) file.
Specific training parameters and the main code are located in [main.py](https://github.com/lischilpp/bachelor-thesis-phoneme-recognition-alignment/blob/main/src/main.py).

### Execution
To run the training and testing process, execute [main.py](https://github.com/lischilpp/bachelor-thesis-phoneme-recognition-alignment/blob/main/src/main.py). Detailed information about the current training is displayed in the terminal and logged in the Lightning logs directory, which can be viewed using TensorBoard for further analysis.

## Main contributions of this work
### CyclicPlateau scheduler
Introduced a scheduler that combines cyclic learning rates with Learning Rate Reduction on Plateau to get the benefits of both techniques.
Cyclic learning rates reduce the risk of getting stuck in poor local minima by exploring a wider range of solutions, while Learning Rate Reduction on Plateau fine-tunes convergence by lowering the learning rate when validation loss stagnates, enabling precise optimization.

### Phoneme-boundary weighted loss
Developed a custom variant of the cross-entropy loss function that assigns higher weight to phoneme boundaries, enhancing the model's ability to accurately detect precise transitions between phonemes.
