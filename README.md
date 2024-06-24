# Bachelor Thesis - Phoneme classification and alignment through recognition on TIMIT

My bachelor thesis on Phoneme recognition and alignment on the TIMIT dataset.

 [Link to publication](https://isl.anthropomatik.kit.edu/downloads/BachelorarbeitSchlipp.pdf)

 ## Abstract

In this work we explore a hybrid between ANNs and DTW for phoneme alignment on the TIMIT dataset. The idea is to use the output probabilities of a neural phoneme recognition model together with a probability-based DTW in order to align phonemes.
For phoneme recognition we achieve 18.1% FER which is an 4.0% improvement over the state- of-the-art.
Our alignment results in a 86.3% phoneme boundary accuracy with a 20ms tolerance. Furthermore phoneme classification based on recordings of single phonemes is being tried resulting in an accuracy of 66.68%.
Apart from that we introduce the CyclicPlateauScheduler, a new learning rate scheduler combining triangular cyclic learning rates with ReduceLROnPlateau.
