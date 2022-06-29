# Customized VAE for computing texture embeddings
A tweaked VAE suitable for computing embeddings for relatively high resolution textures (e.g. of materials).

Following techniques are employed to enhance the performance of the VAE:
- Since textures tend to have recurring patterns, the model can focus on the details by only looking at small patches of the image. This allows the input dimension of the network to be kept small when dealing with high resolution textures.
  - During training, the model is fed randomly augmented and cropped samples.
  - To compute an embedding, the model is fed *n* evenly spaced patches of the input. Then, the average of the embeddings is used.
- Contrast enhancement is applied to the input, allowing the model to better recognize texture details.
- Similar to SimCRL, the difference in embeddings of two randomly augmented versions of the same image are minimized during training, in addition to the usual VAE loss funciton.

Empirically, visually similar textures appear to yield embeddings close to one another, which is in line with human perception.

## Prerequisites
- Python 3.7
- TensorFlow 2.7
- numpy
- PIL
- pickle
