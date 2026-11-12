### Coding Assignment 1: Convolutional Autoencoder

Task: Implement a convolutional autoencoder (CAE) using PyTorch to reconstruct MNIST digits.

Requirements:

Load MNIST data (normalize to [0,1])
Encoder: Conv2d(1→16, 3×3, stride=2, padding=1) → ReLU → Conv2d(16→32, 3×3, stride=2, padding=1) → ReLU → Flatten → Linear(32×7×7 → 20)
Decoder: Linear(20 → 32×7×7) → Reshape → ConvTranspose2d(32→16, 3×3, stride=2, padding=1, output_padding=1) → ReLU → ConvTranspose2d(16→1, 3×3, stride=2, padding=1, output_padding=1) → Sigmoid
Loss: MSE between input and reconstruction
Train with Adam (lr=1e-3, batch size 128, 20 epochs)
Report test MSE and visualize 8 original vs. reconstructed digit pairs
Deliverable: Python script with training loop, test MSE, and a matplotlib grid of reconstructions.  

Coding Assignment 2: Denoising Convolutional Autoencoder

Task: Extend Assignment 1 into a Denoising CAE (DCAE) that reconstructs clean images from noisy inputs.

Requirements:

Same encoder/decoder architecture as Assignment 1
During training, corrupt inputs with Gaussian noise (σ = 0.3) and train to reconstruct the clean target
Evaluate on test set at multiple noise levels: σ ∈ {0.1, 0.3, 0.5}
Compare test MSE against a vanilla CAE (trained on clean data, tested on noisy data)
Produce a side-by-side visualization: clean | noisy | DCAE reconstruction
Deliverable: Python script with noise-level comparison table and visualization grid.

Coding Assignment 3: Convolutional Autoencoder vs. PCA on Images

Task: Compare a convolutional autoencoder against PCA for image compression.

Requirements:

Use MNIST or Fashion-MNIST
Train a CAE with latent dimension k = 10 (same architecture as Assignment 1, adjust final Linear)
Apply PCA to flattened images, retaining k = 10 components
For both methods, compute:
Reconstruction MSE on test set
Visual reconstruction of 5 sample digits
Compare: which method preserves spatial structure better? Which is faster to train/infer?
Deliverable: Python script with MSE comparison, side-by-side reconstruction grid, and a short written analysis (3–5 sentences).