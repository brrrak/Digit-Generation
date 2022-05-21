CMPE 597 Homework 2
Author: Ali Burak Acar

Google Drive Folder for pretrained weights:
https://drive.google.com/drive/folders/11R-S7gslcECuID0_YhLc7iUCbczoXGlu?usp=sharing

This link has two folders in it:
---| ImgsDuringTraining
---| model_checkpoints

model_checkpoints folder should be downloaded and placed in the same folder as the .py files.

Optionally, in the ImgsDuringTraining folder, you can see how training progressed for GAN models.
I have generated pictures from a fixed noise for each epoch and made the results into an animation.
After downloading them, when you click on an html file it should open in a web browser and show you the video.

For both of the models, training code starts from random weights while evaluation code needs the pretrained weights I have provided.

----------VAE Model----------
How to run training code for 5 epochs:
> python3 vaeTrain.py --epochs 5
By default it will visualize the loss.
If you want to turn it off you can add --visualize 0

How to run evaluation code:
> python3 vaeEval.py



----------GAN Model----------

# Training Code:

There are 4 different GAN models.
You can switch between them with "--complex" and "--wgan" arguments.
As the names imply, "--complex" decides whether to use the simpler or more complex generator.
"--wgan" decides whether to use Wasserstein loss or cross-entropy loss.

So you can try the following combinations:

# Simple DCGAN
> python3 ganTrain.py --epochs 1 --complex 0 --wgan 0

# Complex DCGAN
> python3 ganTrain.py --epochs 1 --complex 1 --wgan 0

# Simple WGAN
> python3 ganTrain.py --epochs 1 --complex 0 --wgan 1

# Complex WGAN
> python3 ganTrain.py --epochs 1 --complex 1 --wgan 1

Again, loss visualization is enabled by default, which you can turn off with --visualize 0.

# Evaluation Code:

Again you can choose from the 4 possible models:

# Simple DCGAN
> python3 ganEval.py --complex 0 --wgan 0

# Complex DCGAN
> python3 ganEval.py --complex 1 --wgan 0

# Simple WGAN
> python3 ganEval.py --complex 0 --wgan 1

# Complex WGAN
> python3 ganEval.py --complex 1 --wgan 1


-----------------------------

Both evaluation files also have an argument called --generateForFID which is used to calculate FID and KID scores in my report.
I suggest you don't use this option as it needs mnist in png format and takes 1 hour to calculate per model with GPU.
Nevertheless, I will provide how to use this for documentation purposes.

We need the clean-fid package:
pip install clean-fid

Download and unzip mnist in png format provided from this link:
https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz

Place the unzipped folder called mnist_png in the same folder with .py files.

Run the evaluation code of the desired model with the flag "--generateForFID 1".

It will first create statistics for mnist from the png pictures. (One time only)
Then will generate and save 60,000 images from the chosen model.
Afterwards, it will calculate FID and KID from the generated pictures based on mnist statistics.









