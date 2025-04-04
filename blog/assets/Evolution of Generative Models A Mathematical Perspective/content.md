# Evolution of Generative Models in Computer Vision: A Mathematical Perspective

Published Date: March 23, 2025
Tags: Computer Vision, Machine Learning, Generative Models, Featured

## Introduction

If we asked a computer program would compute driving a car and composing a song 40 years ago, it would be impossible. But in 2024, maybe we can't fully perform self driving car, but we can have a composed song fairly good from a computer program. For most of the history, computers were seen as logical machines to produce rigid solutions, leaving no place for creativity or ambiguity. But in the shift to Generative AI, we can asked it to create and produce something that never exist before. How did this happen? What to promote this shift?

Generative models have revolutionized computer vision by enabling systems not only to recognize and label images but to create them. These models learn to approximate complex high-dimensional data distributions, allowing them to generate novel samples that resemble real-world data. The applications are range from art synthesis and medical imaging to simulation, data augmentation, and inverse problems. 

This article presents a mathematically grounded survey of the evolution of generative models in computer vision. We also explore the history of generative models and what the math intuition behind of this. We trace their theoretical roots and practical transformations, from early probabilistic methods to today’s powerful neural architectures.

## Modeling Generative Models

Before we dive in to generative models, let's talk about discriminative models first. Say that we were given a dataset of pairs $\mathcal{D}={(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}$. In high-level understanding, the goal for this model is to find a model of the conditional probability $p(y|x)$. For example, in image classification such as cat or dog classification, we would want to compute the probility of a cat $(Y='Cat')$ given an image ($X$), formally is written as $p(Y='Cat'|X)$. This model outputs the same probability everytime it runs.

On contrast, generative models are designed to learn the underlying distribution of a dataset in order to generate new data points that are statistically similar to those in the training set. In simpler terms, instead of classifying images from cat and dog datasets, generative models should be able to generate new, never-before-seen images that look like cats and dogs.

Formally, given a dataset $X$, generative models try to model $P(X)$ or the joint distribution $P(X,Y)$ if labels are involved. Figure 1 represents how the generative models are trained and sampled. But how the generative models can come to here? In the boom of generative AI, it is easy to forget that this revolution began from a simple question: *what if we introduce uncertainty in models?*


## Boltzmann Machine as The First Generative Model

The Botzmann Machine introduced a radical notion in 1985:  What if, instead of storing rigid facts and performing deterministic computations, we introduce uncertainty and randomness into neural networks? Before going deeper, let's talk about the predecessor networks, **Hopfield Networks** by John Hopfield (1982). This network can recall and recover data based on its pattern. However, it only completes the data every run. It doesn't introduce a level of uncertain to the network. For example, if there were a musician given a cue to play a song, they would complete the song same everytime. In contrast, Boltzmann Machine introduced several possibilities of how the song can be composed by adding some uncertainty.

Boltzmann machine is a stochastic version of the Hopfield network. It defines a probability distribution over binary vectors using the Boltzmann distribution. Ludwig Boltzmann was a physicist who made profound contributions to statistical thermodynamixs. The question was simple, *if you have macroscopic properties of systems made up a large number of particles, how would you find a state of energy in every particles?* The answer would be *you can't*. but you can define a **probability distribution** over all possible energy states $E$. This gave rise o the **Boltzmann distribution**:

$$P(E) \propto \mathcal{exp}(- \dfrac{E}{k_BT})$$

where $E$ is the energy state, $T$ is the absolute temperature, and $k_B$ is Boltzmann's contsant. From this equation, we can see that the lower energy, the higher the probability. But it doesn't answer the absolute value of $P(E)$. Suppose that energy has many levels, and we want to compare between $E_1, E_2, E_3,...,E_N$. We only know that $P(E_1)+P(E_2)+P(E_3)+...+P(E_N)=1$. We can add the normalization for each absolute value with $Z$, where
$$P(E) \propto \dfrac{1}{Z}\mathcal{exp}(- \dfrac{E}{k_BT})$$
$$Z=\Sum_s \mathcal{exp}(-E_s/k_BT)$$

And so a network, we want to know whether a node is on or off. In a Hopfield Network, the network minimizes an energy function to reach a stable configuration (a memory). The energy of a binary state vector \( \mathbf{x} \in \{-1, +1\}^n \) is:

$$E(x)=- \sum_{i,j} w_(i,j)x_ix_j $$

where the nodes on/off is determined by whether the energy is positive or negative.

The neurons update **deterministically**: each neuron flips based on the sign of the total input from other neurons.

Now, let's make the system **stochastic** — and allow it to **sample** from a distribution over possible states.

The **Boltzmann Machine** defines the probability of a state \( \mathbf{x} \) as:

$$
P(\mathbf{x}) = \frac{1}{Z} \exp\left(-E(\mathbf{x})\right)
$$

Where the energy function is:

$$
E(\mathbf{x}) = -\sum_{i < j} w_{ij} x_i x_j - \sum_i b_i x_i
$$

And \( x_i \in \{0, 1\} \). Note that in Boltzmann Machines, we typically use **binary (0/1)** units instead of \(-1/+1\).


Unlike Hopfield’s deterministic update rule, each neuron in a Boltzmann Machine is turned **on or off stochastically** according to:

$$
P(x_i = 1 | \text{rest}) = \sigma\left( \sum_j w_{ij} x_j + b_i \right)
$$

Where \( \sigma(z) = \frac{1}{1 + \exp(-z)} \) is the sigmoid function.

This allows the network to **explore many configurations**, not just settle in a fixed one.

From here, why stochasticity matters in network learning? Firstly, it allows models to escape local minima, where the randomness helps to escape poor local optima. In Hopfield network, once the models land in a local minima, it stays there--even if that state is not optimal. Secondly, it allows models to sample uncertainty rather to compute a single outcome.



## Histoical Overview of Generative Models

Generative models are never far from Classical (Bayesian) statistics that is a theory based on a degree of belief in an event. The degree of belief may based on prior knowledge about the event as opposed to the frequentist interpretation, which views probability as the relative frequency of an event. One of the methods is using Gaussian Mixture Model. This model is basically tries to fit the data set with a model composed from several Gaussians, each identified by $k\in{1,...,K}$, where $K$ is the number of clusters of the dataset. This is represented in Figure 2. Each Gaussian $k$ in the function is comprised of the following parameters:
- A mean $\nu$ that defines its center.
- A covariance $\Sigma$ that defines its width.
- A probability $\pi$ that defines how big or small the Gaussian function will be compared to other Gaussians.

$$\Sum_{k=1}^K\pi_k=1$$

If we are able to train this model with Gaussians on given dataset, we can generate new data based on the optimal model representing the given dataset.

Many of the first neural nets (1980s-1990s) were generative models. Restricted Boltzmann Machine is one of them, which is an artificial neural networks that capable of learning a probability distribution over a set of input data.

However, up until 2010s, there were deep learning revolution that made discriminative models dominated AI research fields. Not up until 2013, Kingma & Welling introduce Variational Autoencoder.

## Variational Autoencoder

Before going deeper on Varitional Autoencoder (VAE), let's talk about what Autoencoders (AE) are. A very odd idea of it is how about we train a neural network of $f_{\theta}$ to predict the input ($x$) from the input ($x$) itself as in the Figure 3? $(f_\theta(\mathbf{x}) \approx \mathbf{x})$. It would not be useful if we just have the neural nets with only 2 layers of the same width, right? The model won't learn anything. Instead, we can make the neural nets to be deeper and deeper up until the latent layers ($d_{latent}$). This network become into two parts:
- The **encoder** \( E_\phi(\mathbf{x}) \) compresses the input into a lower-dimensional **latent representation** \( \mathbf{z} \in \mathbb{R}^{d_{\text{latent}}} \)
- The **decoder** \( D_\theta(\mathbf{z}) \) attempts to reconstruct the original input from this latent code.


This works only if we **restrict the capacity** of the network — typically by introducing a "bottleneck" (i.e., \( d_{\text{latent}} \ll d_{\text{input}} \)), which forces the model to learn **compact representations** that capture the essential structure of the data.

The intuition would go like this. Imagine you're trying to describe a car to someone over the phone, but you’re only allowed to use three words. You might say "red sports car." These three words are a compressed **latent code**. If the person on the other end is smart enough (your decoder), they could reconstruct a mental image that’s **close** to the original car, even if not exact. But standard autoencoders have a limitation: they don't tell you how **likely** a given car is, nor can they **sample new plausible cars**. The latent space is just a collection of points — there's no structure or distribution. It means that while AEs can **compress and reconstruct data**, they don’t model a probability distribution over the data. There's no clear way to sample *new* data — there's no \( p(\mathbf{x}) \) being learned. This is where **Variational Autoencoders** come in.

VAEs, proposed by **Kingma and Welling (2013)**, address this limitation by combining the representational power of autoencoders with **probabilistic modeling**. Instead of learning fixed encodings, they learn a **distribution over latent variables**.

The core idea is if we assume that data \( \mathbf{x} \) is generated by a **latent variable model**,

\[
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}
\]

Here:
- \( \mathbf{z} \) is a latent vector sampled from a prior distribution (usually \( \mathcal{N}(0, I) \))
- \( p_\theta(\mathbf{x} | \mathbf{z}) \) is a decoder that tells us how to generate \( \mathbf{x} \) given \( \mathbf{z} \)

this defines a **generative process**. But learning this model — i.e., finding the parameters \( \theta \) that maximize \( p_\theta(\mathbf{x}) \) — is hard, because computing the marginal likelihood involves integrating over all \( \mathbf{z} \), which is usually intractable.

To solve this, we introduce a **variational approximation** to the true posterior \( p_\theta(\mathbf{z} | \mathbf{x}) \). We define an encoder network \( q_\phi(\mathbf{z} | \mathbf{x}) \) that approximates it. This gives us a way to "invert" the generative process and map data back into the latent space. This works using KL Divergence: it is measures how much information is lost when using $q_\phi$ to approximate $p_\theta$

We then optimize the **Evidence Lower Bound (ELBO)**:

\[
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
\]

This has two parts:
- The **reconstruction term**: how well can we decode \( \mathbf{x} \) from the sampled \( \mathbf{z} \)?
- The **KL divergence**: how close is the approximate posterior to the prior?

However, there’s a challenge: how can we backpropagate through a **random sample** \( \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x}) \)?

VAEs solve this using the **reparameterization trick**. Instead of sampling \( \mathbf{z} \) directly, we sample \( \boldsymbol{\epsilon} \sim \mathcal{N}(0, I) \) and compute:

\[
\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}
\]

This makes the sampling **differentiable**, so we can train the encoder and decoder end-to-end using stochastic gradient descent.

You can play the visualization of VAE on this link.


## Representation Learning

## Diffusion Models: Generating Images by Learning to Denoise

## Diffusion Models

In generative modeling, one of the most fundamental questions is: How can we teach a model to synthesize complex, high-dimensional data — such as realistic images — from scratch **Diffusion models** offer a remarkably simple yet effective answer, *Learn to start from pure noise, and gradually transform it into a coherent image, step by step*.

Unlike GANs, which try to trick a discriminator, or VAEs, which compress and decode data through a latent space, diffusion models take a **direct, iterative approach**: **denoise the data until it looks real**.

Suppose you have a clear photo and you slowly add noise to it — bit by bit — until it becomes indistinguishable from random pixels. Now imagine trying to **undo** that noise. If a neural network could learn how to reverse that process at every step, you could start from random noise and reconstruct a plausible image.

In diffusion models, it starts with adding noise to the real data over many steps (Forward process), and then train a model to denoise step-by-setp until it gets a sample that resembles real data (Reverse process).

### The Forward Process: Gradual Noise Injection

Let \( \mathbf{x}_0 \) be a real data sample (e.g., an image). We define a noising process that adds Gaussian noise in small increments across \( T \) timesteps:

\[
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
\]

Here:
- \( \beta_t \in (0, 1) \) controls how much noise is added at timestep \( t \).
- After many steps, \( \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) \), meaning the sample is nearly pure noise.

The forward process is **fixed** (non-learned), and we can sample from any step \( t \) in closed form:

\[
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
\]

Where:
- \( \alpha_t = 1 - \beta_t \)
- \( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \)

This means we can take a clean image and **simulate its noisy versions** at various timesteps.

### The Reverse Process: Learning to Denoise

Now comes the learning part. We want to learn a **reverse process** that gradually transforms noise back into data. That is, we define a model \( p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \) that tries to approximate the **posterior** of the forward process:

\[
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \approx q(\mathbf{x}_{t-1} | \mathbf{x}_t)
\]

In practice, we model this as a Gaussian:

\[
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
\]

The parameters \( \mu_\theta \) and \( \Sigma_\theta \) are predicted by a neural network (e.g., a U-Net).


For the training objectives, rather than directly modeling the posterior \( q(\mathbf{x}_{t-1} | \mathbf{x}_t) \), we can instead train a model to predict the **noise** \( \boldsymbol{\epsilon} \) added to each sample at each timestep.

Let:

\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
\]

We then train a neural network \( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \) to predict the noise \( \boldsymbol{\epsilon} \). The loss function is simply:

\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
\]

This is called the **denoising score matching loss**, and it is remarkably effective.


Once the model is trained, generation proceeds as follows:

1. Start with \( \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) \)
2. Iteratively apply the reverse model:
   \[
   \mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
   \]
3. After \( T \) steps, you obtain \( \mathbf{x}_0 \): a synthetic data sample.

Despite taking many steps (typically 50–1000), the process produces **extremely high-quality, diverse images**, making this method becomes powerful:
- **Stable training**: No adversarial game like in GANs.
- **Sample diversity**: Can model complex, multimodal distributions.
- **Likelihood estimation**: With modifications, they can compute log-likelihoods (unlike GANs).
- **Unconditional and conditional generation**: They support class conditioning, text prompts (e.g., in Stable Diffusion), and more.


## Generative Adversarial Networks

## Normalising Flows

## Conditioning and Bayesian Inference

## Evaluate Generative Models