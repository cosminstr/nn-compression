## nn-compressor

My attempt at creating a compression tool using neural networks. I trained a deep U-NET to reconstruct the input using the VOC2010 dataset.

The encoder is the "compressor".
The decoder is the "decompressor".
The bottleneck is the "compressed file".

Since my implementation is of a residual U-NET, I have to save the feature maps from before each downsample block
too in order to reconstruct the input. As such, there is not really any compression being done.

I added this on Github because I think intuitively this is what a beginner would think of when trying to use
neural networks for compression, so maybe it helps someone to avoid my naive approach (I only realised at test time that saving
all the feature maps on disk will not result in compression :/).

You would be better of by starting from [here](https://github.com/facebookresearch/NeuralCompression?tab=readme-ov-file).

## original image vs decompressed image

The U-NET is also not doing a good job at reconstructing the input, but the purpose of this project was not
to train a competent model for this task, rather to just try to make a compression tool.

![image](https://github.com/cosminstr/nn-compression/blob/main/imgs/2007_000333.jpg)
![decompressed_image](https://github.com/cosminstr/nn-compression/blob/main/imgs/decompressed_image.jpg)
