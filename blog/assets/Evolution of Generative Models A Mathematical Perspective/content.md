# Evolution of Generative Models in Computer Vision: A Mathematical Perspective

Published Date: March 23, 2025
Tags: Computer Vision, Machine Learning, Generative Models, Featured

## Introduction

Generative models have revolutionized computer vision by enabling systems not only to recognize and label images but to create them. These models learn to approximate complex high-dimensional data distributions, allowing them to generate novel samples that resemble real-world data. Applications range from art synthesis and medical imaging to simulation, data augmentation, and inverse problems.

This article presents a mathematically grounded survey of the evolution of generative models in computer vision. We trace their theoretical roots and practical transformations, from early probabilistic methods to today’s powerful neural architectures.

## Modeling Generative Models

Before we dive in to generative models, let's talk about discriminative models first. Say that we were given a dataset of pairs $\mathcal{D}={(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}$. In high-level understanding, the goal for this model is to find a model of the conditional probability $p(y|x)$. For example, in image classification such as cat or dog classification, we would want to compute the probility of a cat $(Y='Cat')$ given an image ($X$), formally is written as $p(Y='Cat'|X)$. This model outputs the same probability everytime it runs.

On contrast, generative models are designed to learn the underlying distribution of a dataset in order to generate new data points that are statistically similar to those in the training set. In simpler terms, instead of classifying images from cat and dog datasets, generative models should be able to generate new, never-before-seen images that look like cats and dogs.

Formally, given a dataset $X$, generative models try to model $P(X)$ or the joint distribution $P(X,Y)$ if labels are involved. Figure 1 represents how the generative models are trained and sampled. But how the generative models can come to here? And how is the mathematical background of this?

## Histoical Overview of Generative Models

The very first generative models are based from Classical (Bayesian) statistics that is a theory based on a degree of belief in an event. The degree of belief may based on prior knowledge about the event as opposed to the frequentist interpretation, which views probability as the relative frequency of an event. One of the methods is using Gaussian Mixture Model. This model is basically tries to fit the data set with a model composed from several Gaussians, each identified by $k\in{1,...,K}$, where $K$ is the number of clusters of the dataset. This is represented in Figure 2. Each Gaussian $k$ in the function is comprised of the following parameters:
- A mean $\nu$ that defines its center.
- A covariance $\Sigma$ that defines its width.
- A probability $\pi$ that defines how big or small the Gaussian function will be compared to other Gaussians.

$$\Sum_{k=1}^K\pi_k=1$$

If we are able to train this model with Gaussians on given dataset, we can generate new data based on the optimal model representing the given dataset.

Many of the first neural nets (1980s-1990s) were generative models. Restricted Boltzmann Machine is one of them, which is an artificial neural networks that capable of learning a probability distribution over a set of input data.

However, up until 2010s, there were deep learning revolution that made discriminative models dominated AI research fields. Not up until 2013, Kingma & Welling introduce Variational Autoencoder.

## Variational Autoencoder

Before going deeper on Varitional Autoencoder (VAE), let's talk about what Autoencoders are. A very odd idea of it is how about we train a neural network of $f_{\theta}$ to predict the input ($x$) from the input ($x$) itself as in the Figure 3? It would not be useful if we just have the neural nets with only 2 layers of the same width, right? The model won't learn anything. Instead, we can make the neural nets to be deeper and deeper up until the latent layers ($d_{latent}$). This side is called encoder ($E_{\phi}$). From the latent layer, we 

## Representation Learning

## Diffusion Models

## Generative Adversarial Networks

## Normalising Flows

## Conditioning and Bayesian Inference

## Evaluate Generative Models