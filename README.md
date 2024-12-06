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
For phoneme recognition we achieve 18.1% FER which is an 4.0% improvement over the state- of-the-art.
Our alignment results in a 86.3% phoneme boundary accuracy with a 20ms tolerance. Furthermore phoneme classification based on recordings of single phonemes is being tried resulting in an accuracy of 66.68%.
Apart from that we introduce the CyclicPlateauScheduler, a new learning rate scheduler combining triangular cyclic learning rates with ReduceLROnPlateau.

## CNN experiments
The code for the initial CNN experiments can be found [here](https://github.com/lischilpp/bachelor-thesis-phoneme-recognition-alignment_cnn)
