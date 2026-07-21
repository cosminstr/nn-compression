# nn-compressor

An exploration of learned representations and neural compression using a U-NET.

The initial idea behind this project was to write a compression binary using only a NN, instead of
traditional compression algorithms. I wanted to have a "compress" binary which yields the "compressed" file and a "decompress" binary.

The encoder of the network would be the compression binary, the bottleneck would be the compressed file and the decoder would be the decompression binary. A part of the motivation for this project was to get more familiar with exporting and loading PyTorch models, since i had to save the encoders/decoders weights separately after training, and the latent representation as well (as the "compressed" file).

It is basically a practical application of the idea that NNs are in some way compression algorithms.

The model was trained on the VOC2010 dataset to reconstruct images from their learned representations.

## Caveats

Since my implementation is of a residual U-NET, I have to save the feature maps from before each downsample block too in order to reconstruct the input. As such, there is not really any compression being done.

I added this on Github because I think intuitively this is what a beginner would think of when trying to use neural networks for compression, so maybe it helps someone to avoid my naive approach (I only realised at test time that saving all the feature maps on disk will not result in compression :/).

You can find a better starting point for practical neural compression [here](https://github.com/facebookresearch/NeuralCompression).

## Original image vs decompressed image

The U-NET was not optimized for high-quality reconstruction, as the goal of this project was to explore the concept rather than achieve a competitive network for image reconstruction.

![image](https://github.com/cosminstr/nn-compression/blob/main/imgs/2007_000333.jpg)
![decompressed_image](https://github.com/cosminstr/nn-compression/blob/main/imgs/decompressed_image.jpg)
