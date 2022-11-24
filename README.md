# awesome-temporal-point-processes [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
Track of developments in Temporal Point Processes (TPPs)

# Table of Contents
* [Lecture notes, tutorials and blogs](#lecture-notes-tutorials-and-blogs)
* [Papers](#papers)
  * [Survey and review](#survey-and-review)
  * [Recurrent history encoders (RNNs)](#recurrent-history-encoders-rnns)
  * [Set aggregation history encoders (Transformer)](#set-aggregation-history-encoders-transformer)
  * [Continuous time state](#continuous-time-state)
  * [Intensity free](#intensity-free)
  * [Reinforcement learning](#reinforcement-learning)
  * [Noise contrastive learning](#noise-contrastive-learning)
  * [Long range forecasting](#long-range-forecasting)
  * [Neural ordinary differential equation (Neural ODE)](#neural-ordinary-differential-equation-neural-ode)
  * [Counterfactual modeling](#counterfactual-modeling)
  * [Efficient TPPs](#efficient-tpps)
  * [Semi-supervised TPPs](#semi-supervised-tpps)
  * [Intermittent TPPS](#intermittent-tpps)
* [Workshops](#workshops)
* [Books](#books)

  

# Lecture notes, tutorials and blogs 
* [Lecture notes] [Temporal Point Processes](https://courses.mpi-sws.org/hcml-ws18/lectures/TPP.pdf) [2019]
* [Lecture notes] [Temporal Point Processes and the Conditional Intensity Function](https://arxiv.org/pdf/1806.00221.pdf) [arXiv-2018]
* [Lecture notes] [Lectures on the Poisson Process](https://www.math.kit.edu/stoch/~last/seite/lectures_on_the_poisson_process/media/lastpenrose2017.pdf) [2017]

# Papers

## Survey and review
* [Exploring Generative Neural Temporal Point Process](https://arxiv.org/abs/2208.01874) [TMLR-2022] [[Code](https://github.com/bird-tao/gntpp)] 
* [An Empirical Study: Extensive Deep Temporal Point Process](https://arxiv.org/abs/2110.09823v3) [arXiv-2021] [[Code](https://github.com/bird-tao/edtpp)] 
* [Neural Temporal Point Processes: A Review](https://arxiv.org/abs/2104.03528) [IJCAI-2021]

## Recurrent history encoders (RNNs)
* [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690) [NeurIPS-2019] [[Code](https://github.com/omitakahiro/NeuralNetworkPointProcess)]
* [Marked Temporal Dynamics Modeling based on Recurrent Neural Network](https://arxiv.org/abs/1701.03918v1) [PAKDD-2017]
* [Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks](https://arxiv.org/abs/1705.08982) [AAAI-2017] [[Code](https://github.com/xiaoshuai09/recurrent-point-process)]
* **RMTPP**: [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) [KDD-2016] [[Code-TF](https://github.com/musically-ut/tf_rmtpp)][[Code-PyTorch](https://github.com/woshiyyya/ERPP-RMTPP)]




## Set aggregation history encoders (Transformer)
* [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044) [ICLR-2022] [[Code](https://github.com/yangalan123/anhp-andtt)]

## Continuous time state
* **NHP**: [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328) [NeurIPS-2017] [[Code](https://github.com/HMEIatJHU/neurawkes)] 

## Intensity free
* [Wasserstein Learning of Deep Generative Point Process Models](https://arxiv.org/abs/1705.08051) [NeurIPS-2017] [[Code](https://github.com/xiaoshuai09/Wasserstein-Learning-For-Point-Process)]


## Reinforcement learning
* [Learning Temporal Point Processes via Reinforcement Learning](https://arxiv.org/abs/1811.05016) [NeurIPS-2018] [[Code](https://github.com/sli370/Learning-Temporal-Point-Processes-via-Reinforcement-Learning)]
* [Deep Reinforcement Learning of Marked Temporal Point Processes](https://arxiv.org/abs/1805.09360) [NeurIPS-2018] [[Code](https://github.com/Networks-Learning/tpprl)]



## Noise contrastive learning
* [Noise-contrastive estimation for multivariate point processes](https://arxiv.org/abs/2011.00717) [NeurIPS-2020] [[Code](https://github.com/hongyuanmei/nce-mpp)]
* **INITIATOR**: [Noise-contrastive Estimation for Marked Temporal Point Process](https://www.ijcai.org/proceedings/2018/0303.pdf) [IJCAI-2018] 



## Long range forecasting
* **HYPRO**: [A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences](https://arxiv.org/abs/2210.01753) [NeurIPS-2022] [[Code](https://github.com/ilampard/hypro_tpp)]

## Neural ordinary differential equation (Neural ODE)
* [Neural Spatio-Temporal Point Processes](https://arxiv.org/abs/2011.04583) [ICLR-2021] [[Code](https://github.com/facebookresearch/neural_stpp)]
* [Learning Neural Event Functions for Ordinary Differential Equations](https://arxiv.org/abs/2011.03902) [ICLR-2021] [[Code](https://github.com/rtqichen/torchdiffeq)]



## Counterfactual modeling
* [Counterfactual Temporal Point Processes](https://arxiv.org/abs/2111.07603) [NeurIPS-2022] [[Code](https://github.com/Networks-Learning/counterfactual-tpp)]

## Efficient TPPs
* [Fast and Flexible Temporal Point Processes with Triangular Maps](https://arxiv.org/abs/2006.12631v2) [NeurIPS-2020] [[Code](https://github.com/shchur/triangular-tpp)]

## Semi-supervised TPPS
* [Semi-supervised Learning for Marked Temporal Point Processes](https://arxiv.org/abs/2107.07729?context=cs) [IJCAI Workshop-2021]



## Intermittent TPPS
* [Learning Temporal Point Processes with Intermittent Observations](http://proceedings.mlr.press/v130/gupta21a/gupta21a.pdf) [AISTATS-2021] [[Code](https://github.com/data-iitd/imtpp)]


# Workshops
* [Learning with TPPs](https://sites.google.com/view/tpp-neurips-2019) [NeurIPS-2019]

# Books
* [An Introduction to the Theory of Point Processes: Volume I: Elementary Theory and Methods: Daley, D.J., Vere-Jones, D
]
