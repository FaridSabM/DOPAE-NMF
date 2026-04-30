# DOPAE-NMF

This repository provides an implementation for the paper:

DOPAE-NMF: Deep Oblique Projective Autoencoder-Like Non-negative Matrix Factorization For Robust Image Clustering

Yasin Hashemi-Nazari, Farid Saberi-Movahed, Azita Tajaddini, Catarina Moreira

# Description:

The purpose of this paper is to propose a novel variant of deep autoencoder-like NMF, called Deep Oblique Projective Autoencoder-Like NMF (DOPAE-NMF).  The main contributions of this work are summarized as follows:

(1) A novel non-negative matrix factorization model, termed Oblique Projective NMF (OP-NMF), is proposed, in which an oblique projection mechanism is incorporated into the factorization process. By introducing two coupled non-negative matrices that satisfy a bi-orthogonality constraint, the roles of representation and projection are decoupled, enabling more flexible modeling of data structures under complex noise conditions.

(2) Based on OP-NMF, a deep autoencoder-like architecture, referred to as Deep Oblique Projective Autoencoder-Like NMF (DOPAE-NMF), is established. In this formulation, the oblique projective structure is embedded within both the encoding and decoding transformations, enabling progressive refinement of latent representations while enforcing projection consistency across layers.

(3) To further enhance robustness, an adaptive feature-weighting strategy is introduced via two learnable diagonal matrices embedded in the encoder and decoder. This mechanism dynamically adjusts feature importance by suppressing noisy components and emphasizing informative structures throughout the learning process.

(4) In addition, a comprehensive regularization framework is incorporated into the proposed model, consisting of a weighted $L_{1,1}$ sparsity term, contrastive graph-based constraints, and an independence-promoting regularizer. These components are jointly embedded within the final objective function of DOPAE-NMF and collectively preserve local and global data structures, enhance discriminative representation learning, and reduce redundancy among the learned basis vectors.

(5) To the best of our knowledge, this work is the first to present a non-negative formulation of oblique projection within an NMF framework. This contribution provides a new perspective beyond traditional orthogonal projection-based methods by allowing projections to be performed along directions that are not restricted to be orthogonal to the signal subspace. As a result, the model can more effectively isolate meaningful data components while mitigating the influence of structured or correlated noise that may lie within or overlap with the signal subspace.
	
# Citation

If you find this work useful in your research, please consider citing:

Y. Hashemi-Nazari, F. Saberi-Movahed, A. Tajaddini, C. Catarina Moreira, Deep Oblique Projective Autoencoder-Like Non-negative Matrix Factorization For Robust Image Clustering, Expert Systems with Applications, 2026.

# A quick start:

This codebase has been implemented in Python (2026). To run the project, simply execute the file main.py and follow the printed outputs in the console.

The project structure is organized as follows:

main.py – Entry point of the program.

DOPAE‑NMF.py – Core model implementation.

Preprocessing.py – Data preparation and preprocessing utilities.

Libraries.py – Supporting functions and shared utilities.

# Contact

If you have any questions about the method, the implementation, or potential research collaborations, feel free to contact us.

Farid Saberi-Movahed

Email: f.saberimovahed@kgut.ac.ir; fdsaberi@gmail.com
