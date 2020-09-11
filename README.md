# DIF

This code repository applies DIF to IntroVAE: ["IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis"](http://papers.nips.cc/paper/7291-introvae-introspective-variational-autoencoders-for-photographic-image-synthesis)

This code is based on the original implementation of IntroVAE: https://github.com/hhb072/IntroVAE

## Prerequisites
* Python 3.7
* PyTorch>=1.6 (requires mixed precision training)

## Run

To train the model and the benchmarks run the lines of code in run_256.sh for each dataset. A RTX2080Ti card or stronger is recommended.

Make sure to modify the "dataroot" parameter for image source and "class_indicator_file" for labels.

The public datasets can be downloaded by installing "gdown" (i.e. "pip install gdown") and calling:

CelebHQ: gdown https://drive.google.com/uc?id=19WmAelyLp8TA8bAnLdS6oPP1b3z7wlaz \
Fashion: gdown https://drive.google.com/uc?id=1Mxn3Jf1uYozgNa1AAXNXHLzaL6tfQpTI \
MNIST: gdown https://drive.google.com/uc?id=12b-PTIKoCOOUNBMq0Y7B3itUtkJ26ebC \
COVID-19: gdown https://drive.google.com/uc?id=1kpjRm-KWWiffaOYbULsKLIYYJPflmaBx 

These datasets are already preprocessed and can be trained on directly after unzipping them.

They contain both "dataroot" images and the "class_indicator" file.

To post process and get images and fit a lasso model, run the "post_processing_script.py".

Depending on the name of saved models, some tinkering might be needed. 

"progressive_UMAP_plot.py" visualizes the latent space using UMAP in 2-d for different epochs.

"witness_UMAP_plot.py" plots the prototypes in the umap together with latent representations.

"post_post_processing_script.py" outputs every numerical result to latex.

