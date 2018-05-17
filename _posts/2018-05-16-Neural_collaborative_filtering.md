---
layout: post
title: Deep Learning for Recommender Systems: Neural Collaborative Filtering
---

A recommender system is a machine learning model, which predicts the preference a user would assign to an item.  These systems are used to understand what facebook articles you'd like to read, which youtube videos you will watch, what amazon items you'll most likely buy and so on.

I did not used to think recommender systems were particularly interesting.  I remember briefly covering matrix factorization and collaborative filtering (two established recommender techniques) during the Master of Data Science program at the University of British Columbia.  My initial impression was that there must be far more interesting problems to solve than selling an extra Amazon item to a user or predicting which youtube videos someone will view.  It all seemed depressing and vacuous when compared to ML applications in self driving cars, medical image processing and natural language understanding.

However, being fascinated by all applications of deep learning, I've recently been thinking "I wonder if deep learning is used in state of the art recommenders?".  Turns out, yes. Everywhere....

# What is the current state of art?

If you want to read up on the current state of the art the following paper provides a great review:
 
- [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf)
 
And a few must reads:

- [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)
- [Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf)
- [wide and deep learning for recommender systems](https://arxiv.org/pdf/1606.07792.pdf)
- [A Neural Autoregressive Approach to Collaborative Filtering](https://arxiv.org/pdf/1605.09477.pdf)
- [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- [Restricted Boltzmann Machines for Collaborative Filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)

And here are a few great videos:

- [Alexandros Karatzoglou: Deep Learning for Recommender Systems](https://www.youtube.com/watch?v=KZ7bcfYGuxw)
- [Deep Learning for Personalized Search and Recommender Systems part 1](https://www.youtube.com/watch?v=0DYQzZp68ok&t=4999s)

# Ok but why are they interesting?

High quality recommenders typically ingest a **massive quantity and diversity of data**.  Since deep learning attempts to learn feature representations, the more user and product information we input to the model the better. This includes product images, user search queries, click history, purchase sequences and just about everything you can find on a person from their online footprint. As a result state of the art systems utilize many deep learning techniques, in order to ingest this variety of information. Convolutional layers for image ingestion, recurrent layers for ingesting sequential data such as text or click history, autoencoders for denoising input data. The list goes on.  

Since recommenders often have a vast quantity of training data, **very large models can be trained**.  Alexandros Karatzoglou, Scientific Director at Telefonica Research, noted in his lecture [Deep Learning for Recommender Systems](https://www.youtube.com/watch?v=KZ7bcfYGuxw) that their research team train models as large as they can fit in the memory of a single gpu. With the recent advances in gpu hardware, this means we are talking a model with so many parameters it is over 500Gb. Now we have to distribute the training of models on many gpu's in the cloud, and keep prediction time low enough that we can use the model when a user visits our website.

Another reason recommender systems are interesting to me is that many companies make significant online revenue, but are not up to speed on the state of the art.  The potential for a small team of data scientists to increase online sales revenue is an exciting impact to have.

# Neural collaborative filtering

In this post I will cover [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf). This is a very simple model, which provides a great framework to explain our input data, evaluation metrics and some common tricks to deal with scalability problems.

### A note on matrix factorization

Xiangnan He et al. put it best:

> The key to
a personalized recommender system is in modelling users’
preference on items based on their past interactions (e.g.,
ratings and clicks), known as collaborative filtering [31, 46].
Among the various collaborative filtering techniques, matrix
factorization (MF) [14, 21] is the most popular one, which
projects users and items into a shared latent space, using
a vector of latent features to represent a user or an item.
Thereafter a user’s interaction on an item is modelled as the
inner product of their latent vectors.

![](/images/mf.svg)
> Images from [Factorization Machines A Theoretical Introduction](http://jxieeducation.com/2016-06-26/Factorization-Machines-A-Theoretical-Introduction/)

The figure above shows a simple matrix factorization. The ratings matrix  shows the rating each user assigned to each item. Each user (A, B, C, D) is assigned a latent feature vector of size 2. Each item (W, X, Y, Z) is also assigned a latent feature vector of size 2. These vectors are learned, such that their inner product approximates the rating matrix.

This allows us to learn the latent features of items and users. The result is that we can predict the rating a user will assign to an item they have not rated.

Neural collaborative filtering is a flexible neural network architecture, which can be used to learn more complex (non linear) interactions between user and item latent feature vectors. It can also be generalized to represent the traditional matrix factorization method above.

Before I explain the architecture, lets talk more about our input data which will be fed through our neural network.

### Input Data

The raw training data for this model consists of 1,000,000 movie ratings, between 1 & 5. Each record describes the rating a user assigned to an item. There are 943 users and 1682  users in this dataset.

Notice that the raw data is in the form of implicit feedback (reviews from 1-5). This  means a user has actually inferred their preference on an item. In real world scenarios, it is far easier to collect vast quantities of implicit feedback. Clicking on products, buying items, watching movies you get the idea. 

For this reason we will convert the movie reviews into implicit form. We want to build a model that is useful with implicit data. Any movie that has been reviewed gets a 1, any movie that wasn't reviewed gets a 0. 

![](/images/raw_data.png)

We have 6040 users and 3706 movies. This is a total of 22,384,240 possible interactions! 1,000,000 of these combinations will be 1, where the movie was reviewed. The remaining 21,384,240 will be 0, where the users did not review the movies. This is where scalability becomes a problem, since the number of training examples explodes with more users or items.

We do not want to train a model on 22 million input examples. It will take too long. To deal with this problem, negative sampling is used. For the training set, we select *n* random non-interactions (0's) for every interaction in our data. In this post we will sample 4 negatives per interation in the training set (as suggested in the original paper).

For the test set, we take the latest interaction for each user. We then sample  99 negatives for each user. This will allow us to compute meaningful metrics when evaluating our model (more later).

The code below defines a custom MXNet iterator class, which stores the data in the scipy sparse matrix, producing arrays of training data when the batch is required.

### Model Architecture

The general architecture is shown below. Each user & item is assigned a latent vector. This is essentially a list of numbers that numerically represents that user or item. The more similar two user vectors, the more similar those users are. The same for items. 

Each training record consists of a user and item. We simply retrieve the user latent vector, item latent vector and concatenate them. This concatenated vector is passed through three fully connected layers, before being mapped to a single neuron, with an activation function ensuring the model outputs values between 0 and 1. The loss for that record is computed by comparing the model output to the true label (0 or 1).

![](/images/ncf.png)
> Image from [neural collaborative filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) figure 2

### Using the Model

This is a ranking model. We want to know which movies a user is most likely to rate, in real time. To do this, for the user we obtain the model prediction for all items. We then select the k largest values in the list as the items that user is most likely to review.

### Evaluation Metrics

Remember our test set consists of 1 positive interaction and 99 negatives per user? To evaluate the model during training, we obtain the model prediction for all test examples. We then order the predictions for each user. Our metric is computed as the percentage of the time the true interaction falls within the top k ranked items for each user. This is known as hit rate at K.

HR@5=0.8 can be interpreted as 80% of the time the true interaction is within the 5 recommendations we made. This gives an intuition of how this model may perform if deployed.

### Code

A full implementation in Apache MXNet can be found in [my github repo](https://github.com/opringle/collaborative_filtering).

# About the author

>[Oliver Pringle](https://www.linkedin.com/in/oliverpringle/) graduated from the UBC Master of Data Science Program in 2017 and is currently a Data Scientist at [Finn.ai](http://finn.ai/) working on AI driven conversational assistants.

