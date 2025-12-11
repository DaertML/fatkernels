# fatkernels
![fatkernel](fatkernel.png)

# Introduction
Collection of fat kernels for deep learning writen in triton.
Looking forward accelerating the neural networks that we use in our day to day basis; by going beyond optimizations given by DL frameworks like pytorch: e.g.: by fusing kernels, and making big fat kernels that run on NVIDIA GPUs.

# Use
This repo provides code that will resemble into modular components, it follows the ideas of scratchml (https://github.com/DaertML/scratchml), but instead of using CPU and bare python without dependencies, using triton.

This opens up the door to run the same kind of AI models, into accelerators like GPUs, in a simple and non-bloated way.

You can find the optimal parameters to run the LLaMA3.2 model by using "python3 models/search_llama32.py", you can also find optimal ways of running an MLP by using "python3 models/search_mlp.py".

Finally a simple script to show how to perform training of models is available at models/train_perceptron.py.
