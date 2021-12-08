# ArmorSets-GAN

In this repository, I tried to create a Generative Adversarial Network (GAN) for generation of World of Warcraft images. On the images, there shall be new, not yet seen, so called "transmogs" for Zandalari Trolls race. I had this idea in my head for quite some time, and I finally took some time to spend it on this project. Let's get started.

# TO BE DONE
1) RGB images - parametrize definitions of partial models with third channel (in progress on branch "enable_rgb")
2) Finish README
3) Clean the repo, unify img folders etc
4) Resume training function

## Introduction
In WoW, "transmog" is basically a "fashion style" that you apply to your character. Wearing different types of armour can (but don't have to) change your looks. There are specifically created "transmog sets" by the developers, which are usually intended to be well-matched together. Players, of course, experiment and often create nice and unique transmogs by combining several other transmog sets together. Note that a given item may or may not be a part of a transmog set. Since the process of creating a new, nice looking outfit needs a bit of creativity and can be considered an art to some extent, why not use GANs to generate this art? I will not be creating combinations of existing transmogs per se, rather the network shall learn new pieces of armor that could potentially be added to the game.

## Dataset
As with other machine learning problems, the most complicated part is often the data. For this project, I sat down and created my own dataset from scratch. 
The main source of images was [WoWHead website](https://www.wowhead.com/). More specifically, I used their "outfits" section, where I applied a filter too look only for outfits that can be displayed on [**Zandalari Trolls**](https://www.wowhead.com/outfits/race:31/gender:0). All of these outfits were created by players (They are not sets per se), so most of them are indeed unique. For the sake of simplicity, I considered only **Male Zandalari Trolls**. This website has a great potential for expanding the training dataset for other races, as well as other genders. I will be updating the dataset from time to time, whenever there is a good amount of new training examples on the website.

After applying this filter, I went through all the pages, displaying 1000 results of Zandalari Trolls outfits. From these 1000 results, I have omitted ie. sets which had only one piece of armor applied (ie. only boots on the whole character). For all other outfits displayed in the filter, I have opened the model in WoWHead's 3D viewer, and used a simple PrintScreen (With a little help of Lightshot) to save the posture. All the models were adjusted in the 3D viewer to have a better look from the front. Since all has been done by hand, the position is not always the same, but that can never hurt a learning neural network :). This way, I have created a dataset of about 650 pure Zandalari Troll transmogs. Afterwards, I have also applied the same data generating process on transmog sets, which are named in the dataset as `<set_name>_ZT.png`. You may notice, that in the "dataset" folder / file, there are also images ending in `_G` and `_H`. These are again transmog sets, but this time on Gnomes and Humans. This is a relict of mz initial enthusiasm and idea that I would generate this dataset for Zandalari Trolls as well as Gnomes and Humans. (But Gnomes and Humans are also used in the current state of the network, TBD)

During the genration of the dataset, I noticed that a transmog item `Xavius' Shoulders` were not displayed in the 3D viewer. Because of that, I have added a few images with the character wearing this item from different sources than WoWHead database.

All in all, there are 971 training images of transmog sets, with 895 of them being Zandalari Trolls.

An example image from the dataset is the following. 

![](img/test1.png) ![](img/test1_gray.png)

The first version of the dataset is available for [download](https://drive.google.com/file/d/1qCJO6fglDQ8qKJPVduYdcdQuxru9PDwP/view?usp=sharing)

### Why Zandalari Trolls?
In my **subjective** opinion, Zandalari Trolls and Humans have the best body posture. If you are familiar with the races, for example Trolls, or Orcs are usually stooped and look unnatural to me. Zandalari Trolls are also more muscular and greater in size then Humans. As they are the true alphas of the game, I decided to show my love for them this way.

## Environment
- Tensorflow 2.7.0 
- tqdm 4.31.1
- opencv-python 4.5.4
- CUDA 11.5.1_496.13
- cuDNN v8.2.2 (July 6th, 2021), for CUDA 11.4 (cuDNN for 11.5 does not work)
- NVIDIA GeForce GTX 1060 & NVIDIA GeForce RTX 2080

It is recommended to use GPU or any other accelerator for training the network. Generation can probably be done on CPU, though expect it won't be instantly. For a guide on how to install tensorflow for GPU, refer to [tensorflow-gpu installation guide](https://www.youtube.com/watch?v=hHWkvEcDBO0&ab_channel=AladdinPersson). The official documentation on how to install GPU support for tensorflow is not clear.

## Results
The results obtained look very promising. TBD update with more epochs trained. The model learns to generate a posture similar to Zandalari Trolls. An interesting, yet not suprising, fact is, that the network has learned to generate also the WoWHead logo in the background, which was present in most of training images. (TBD add image illustrating this). As far as I know, there is no rigorous metric to evaluate the performance of GANs, so a human evaluator needs to assess the quality of generated images. I will update this section when I get results from ie 10.000 epochs.

## Sources
- [WoWHead website](https://www.wowhead.com/)
- [How to develop a generative adversarial network for an mnist handwritten digits from scratch in Keras](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/)
- [Image generation in 10 minutes with generative adversarial networks](https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b)
- [Tensorflow on GPU](https://www.tensorflow.org/guide/gpu)
