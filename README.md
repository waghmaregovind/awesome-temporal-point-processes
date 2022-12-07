# awesome-temporal-point-processes [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
Track of developments in Temporal Point Processes (TPPs)

# Table of Contents
* [Lecture notes, tutorials and blogs](#lecture-notes-tutorials-and-blogs)
* [Papers](#papers)
  * [Survey and review](#survey-and-review)
  * [Recurrent history encoders (RNNs)](#recurrent-history-encoders-rnns)
  * [Set aggregation history encoders (Transformer)](#set-aggregation-history-encoders-transformer)
  * [Continuous time state](#continuous-time-state)
  * [Intensity free and likelihood free](#intensity-free-and-likelihood-free)
  * [Conditionally dependent modeling of time and mark](#conditionally-dependent-modeling-of-time-and-mark)
  * [Reinforcement learning](#reinforcement-learning)
  * [Noise contrastive learning](#noise-contrastive-learning)
  * [Long range event forecasting](#long-range-event-forecasting)
  * [Neural ordinary differential equation (Neural ODE)](#neural-ordinary-differential-equation-neural-ode)
  * [Counterfactual modeling](#counterfactual-modeling)
  * [Normalizing flows and Efficient TPPs](#normalizing-flow-and-efficient-tpps)
  * [Intermittent TPPS](#intermittent-tpps)
  * [TPPs and graphs](#tpps-and-graphs)
  * [Hawkes process](#hawkes-process)
  * [Adversarial](#adversarial)
  * [Retrieval](#retrieval)
  * [Others](#others)
* [Workshop papers](#workshop-papers)
* [Workshops](#workshops)
* [Applications](#applications)
* [Books](#books)
* [Other awesome-tpp repos](#other-awesome-tpp-repos)


# Lecture notes, tutorials and blogs 
* Lecture notes 
  * [Temporal Point Processes](https://courses.mpi-sws.org/hcml-ws18/lectures/TPP.pdf) [2019]
  * [Temporal Point Processes and the Conditional Intensity Function](https://arxiv.org/pdf/1806.00221.pdf) [arXiv-2018]
  * [Lectures on the Poisson Process](https://www.math.kit.edu/stoch/~last/seite/lectures_on_the_poisson_process/media/lastpenrose2017.pdf) [2017]
* Tutorial 
  * [Graphical Models Meet Temporal Point Processes](https://soerenwengel.github.io/files/UAI_2022__Graphical_Models_Meet_TPPs.pdf) [UAI-2022]
  * By Thinklab [KDD-2019]
    * [Part-I: Modeling and Applications for TPPs](http://thinklab.sjtu.edu.cn/paper/KDD2019_tpp_tutorial-PartI.pdf)
    * [Part-II: Deep Networks for TPPs](http://thinklab.sjtu.edu.cn/paper/KDD2019_tpp_tutorial-PartII.pdf)
    * [Part-III: TPPs in Practice](http://thinklab.sjtu.edu.cn/paper/KDD2019_tpp_tutorial-PartIII.pdf)
  * [Learning with TPP](https://learning.mpi-sws.org/tpp-icml18/) [ICML-2018]
* Blog 
  * By [Oleksandr Shchur](https://shchur.github.io/)
    * [Temporal Point Processes 1: The Conditional Intensity Function](https://shchur.github.io/blog/2020/tpp1-conditional-intensity/)
    * [Temporal Point Processes 2: Neural TPP Models](https://shchur.github.io/blog/2021/tpp2-neural-tpps/)


# Papers

## Survey and review
* [Exploring Generative Neural Temporal Point Process](https://arxiv.org/abs/2208.01874) [TMLR-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/bird-tao/gntpp?style=social)](https://github.com/bird-tao/gntpp) 
* [An Empirical Study: Extensive Deep Temporal Point Process](https://arxiv.org/abs/2110.09823v3) [arXiv-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/bird-tao/edtpp?style=social)](https://github.com/bird-tao/edtpp)
* [Neural Temporal Point Processes: A Review](https://arxiv.org/abs/2104.03528) [IJCAI-2021]
* [Recent Advance in Temporal Point Process: from Machine Learning Perspective](https://thinklab.sjtu.edu.cn/src/pp_survey.pdf) [2019] 


## Recurrent history encoders (RNNs)
* [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690) [NeurIPS-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/omitakahiro/NeuralNetworkPointProcess?style=social)](https://github.com/omitakahiro/NeuralNetworkPointProcess)
* [Marked Temporal Dynamics Modeling based on Recurrent Neural Network](https://arxiv.org/abs/1701.03918v1) [PAKDD-2017]
* [Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks](https://arxiv.org/abs/1705.08982) [AAAI-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/xiaoshuai09/recurrent-point-process?style=social)](https://github.com/xiaoshuai09/recurrent-point-process)
* **RMTPP**: [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) [KDD-2016] [![GitHub Repo stars](https://img.shields.io/github/stars/musically-ut/tf_rmtpp?style=social)](https://github.com/musically-ut/tf_rmtpp)


## Set aggregation history encoders (Transformer)
* [Transformer Embeddings of Irregularly Spaced Events and Their Participants](https://arxiv.org/abs/2201.00044) [ICLR-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/yangalan123/anhp-andtt?style=social)](https://github.com/yangalan123/anhp-andtt)
* [Deep Fourier Kernel for Self-Attentive Point Processes](https://arxiv.org/abs/2002.07281) [AISTATS-2021]
* **SAHP**: [Self-Attentive Hawkes Processes](https://arxiv.org/abs/1907.07561) [ICML-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/QiangAIResearcher/sahp_repo?style=social)](https://github.com/QiangAIResearcher/sahp_repo)
* **THP**: [Transformer Hawkes Process](https://arxiv.org/abs/2002.09291) [ICML-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/SimiaoZuo/Transformer-Hawkes-Process?style=social)](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)


## Continuous time state
* [User-Dependent Neural Sequence Models for Continuous-Time Event Data](https://arxiv.org/abs/2011.03231) [NeurIPS-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/ajboyd2/vae_mpp?style=social)](https://github.com/ajboyd2/vae_mpp)
* **NHP**: [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328) [NeurIPS-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/HMEIatJHU/neurawkes?style=social)](https://github.com/HMEIatJHU/neurawkes)

## Intensity free and likelihood free
* **LNM**: [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127) [ICLR-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/shchur/ifl-tpp?style=social)](https://github.com/shchur/ifl-tpp)
* [Learning Conditional Generative Models for Temporal Point Processes](https://dl.acm.org/doi/abs/10.5555/3504035.3504807) [AAAI-2018] 
* [Wasserstein Learning of Deep Generative Point Process Models](https://arxiv.org/abs/1705.08051) [NeurIPS-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/xiaoshuai09/Wasserstein-Learning-For-Point-Process?style=social)](https://github.com/xiaoshuai09/Wasserstein-Learning-For-Point-Process)


## Conditionally dependent modeling of time and mark
* [Uncertainty on Asynchronous Time Event Prediction](https://arxiv.org/abs/1911.05503) [NeurIPS-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/sharpenb/Uncertainty-Event-Prediction?style=social)](https://github.com/sharpenb/Uncertainty-Event-Prediction)
* [Neural Temporal Point Processes For Modelling Electronic Health Records](https://arxiv.org/abs/2007.13794) [ML4H at NeurIPS-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/babylonhealth/neuralTPPs?style=social)](https://github.com/babylonhealth/neuralTPPs)
* [Modeling Inter-Dependence Between Time and Mark in Multivariate Temporal Point Processes](https://arxiv.org/abs/2210.15294) [CIKM-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/waghmaregovind/joint_tpp?style=social)](https://github.com/waghmaregovind/joint_tpp)


## Reinforcement learning
* [Bellman Meets Hawkes: Model-Based Reinforcement Learning via Temporal Point Processes](https://arxiv.org/abs/2201.12569) [AAAI-2023] [![GitHub Repo stars](https://img.shields.io/github/stars/Event-Driven-rl/Event-Driven-RL?style=social)
* [Learning Temporal Point Processes via Reinforcement Learning](https://arxiv.org/abs/1811.05016) [NeurIPS-2018] [![GitHub Repo stars](https://img.shields.io/github/stars/sli370/Learning-Temporal-Point-Processes-via-Reinforcement-Learning?style=social)](https://github.com/sli370/Learning-Temporal-Point-Processes-via-Reinforcement-Learning)
* [Deep Reinforcement Learning of Marked Temporal Point Processes](https://arxiv.org/abs/1805.09360) [NeurIPS-2018] [![GitHub Repo stars](https://img.shields.io/github/stars/Networks-Learning/tpprl?style=social)](https://github.com/sNetworks-Learning/tpprl)


## Noise contrastive learning
* [Noise-contrastive estimation for multivariate point processes](https://arxiv.org/abs/2011.00717) [NeurIPS-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/hongyuanmei/nce-mpp?style=social)](https://github.com/hongyuanmei/nce-mpp)
* **INITIATOR**: [Noise-contrastive Estimation for Marked Temporal Point Process](https://www.ijcai.org/proceedings/2018/0303.pdf) [IJCAI-2018] 



## Long range event forecasting
* [Neural multi-event forecasting on spatio-temporal point processes using probabilistically enriched transformers](https://arxiv.org/abs/2211.02922) [arXiv-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/negar-erfanian/neural-spatio-temporal-probabilistic-transformers?style=social)](https://github.com/negar-erfanian/neural-spatio-temporal-probabilistic-transformers)
*  **HYPRO**: [A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences](https://arxiv.org/abs/2210.01753) [NeurIPS-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/ilampard/hypro_tpp?style=social)](https://github.com/ilampard/hypro_tpp)
* **DualTPP**: [Long Horizon Forecasting With Temporal Point Processes](https://arxiv.org/abs/2101.02815) [WSDM-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/pratham16cse/DualTPP?style=social)](https://github.com/pratham16cse/DualTPP)


## Neural ordinary differential equation (Neural ODE)
* [Neural Spatio-Temporal Point Processes](https://arxiv.org/abs/2011.04583) [ICLR-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/neural_stpp?style=social)](https://github.com/facebookresearch/neural_stpp)
* [Learning Neural Event Functions for Ordinary Differential Equations](https://arxiv.org/abs/2011.03902) [ICLR-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/rtqichen/torchdiffeq?style=social)](https://github.com/rtqichen/torchdiffeq)
* [Neural Jump Stochastic Differential Equations](https://arxiv.org/abs/1905.10403) [NeurIPS-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/000Justin000/torchdiffeq?style=social)](https://github.com/000Justin000/torchdiffeq)
* [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907) [NeurIPS-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/YuliaRubanova/latent_ode?style=social)](https://github.com/YuliaRubanova/latent_ode)





## Counterfactual modeling
* [Counterfactual Temporal Point Processes](https://arxiv.org/abs/2111.07603) [NeurIPS-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/Networks-Learning/counterfactual-tpp?style=social)](https://github.com/Networks-Learning/counterfactual-tpp)
* [Causal Inference for Event Pairs in Multivariate Point Processes](https://proceedings.neurips.cc/paper/2021/file/9078f2a8254704bd760460f027072e52-Paper.pdf) [NeurIPS-2021] 
* **CAUSE**: [Learning Granger Causality from Event Sequences using Attribution Methods](https://arxiv.org/abs/2002.07906) [ICML-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/razhangwei/CAUSE?style=social)](https://github.com/razhangwei/CAUSE)
* [Uncovering Causality from Multivariate Hawkes Integrated Cumulants](https://arxiv.org/abs/1607.06333) [ICML-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/achab/nphc?style=social)](https://github.com/achab/nphc)
* [Graphical Modeling for Multivariate Hawkes Processes with Nonparametric Link Functions](https://arxiv.org/abs/1605.06759) [Journal of Time Series Analysis-2017]





## Normalizing flows and Efficient TPPs
* **ProActive**: [Self-Attentive Temporal Point Process Flows for Activity Sequences](https://arxiv.org/abs/2206.05291) [KDD-2022] 
* **TriTPP**: [Fast and Flexible Temporal Point Processes with Triangular Maps](https://arxiv.org/abs/2006.12631v2) [NeurIPS-2020] [![GitHub Repo stars](https://img.shields.io/github/stars/shchur/triangular-tpp?style=social)](https://github.com/shchur/triangular-tpp)
* **FastPoint**: [Scalable Deep Point Processes](https://ecmlpkdd2019.org/downloads/paper/861.pdf) [ECML KDD-2019]
* [Point Process Flows](https://arxiv.org/abs/1910.08281) [arXiv-2019]



## Intermittent TPPS
* [Learning Temporal Point Processes with Intermittent Observations](http://proceedings.mlr.press/v130/gupta21a/gupta21a.pdf) [AISTATS-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/data-iitd/imtpp?style=social)](https://github.com/data-iitd/imtpp)
* [Imputing Missing Events in Continuous-Time Event Streams](https://arxiv.org/abs/1905.05570) [ICML-2019][![GitHub Repo stars](https://img.shields.io/github/stars/hongyuanmei/neural-hawkes-particle-smoothing?style=social)](https://github.com/hongyuanmei/neural-hawkes-particle-smoothing)

## TPPs and graphs
* **TREND**: [TempoRal Event and Node Dynamics for Graph Representation Learning](https://arxiv.org/abs/2203.14303) [WWW-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/WenZhihao666/TREND?style=social)](https://github.com/WenZhihao666/TREND)
* [Mitigating Performance Saturation in Neural Marked Point Processes: Architectures and Loss Functions](https://arxiv.org/abs/2107.03354) [Research track KDD-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/ltz0120/Graph-Convolutional-Hawkes-Processes-GCHP?style=social)](https://github.com/ltz0120/Graph-Convolutional-Hawkes-Processes-GCHP)
* [Learning Neural Point Processes with Latent Graphs](https://dl.acm.org/doi/10.1145/3442381.3450135) [WWW-2021] 
* [Modeling Event Propagation via Graph Biased Temporal Point Process](https://arxiv.org/abs/1908.01623v2) [IEEE Transactions on Neural Networks and Learning Systems-2020]
* **DyRep**: [Learning Representations over Dynamic Graphs](https://openreview.net/forum?id=HyePrhR5KX) [ICLR-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/Harryi0/dyrep_torch?style=social)](https://github.com/Harryi0/dyrep_torch)
* **Know-Evolve**: [Deep Temporal Reasoning for Dynamic Knowledge Graphs](https://arxiv.org/abs/1705.05742) [ICML-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/rstriv/Know-Evolve?style=social)](https://github.com/rstriv/Know-Evolve)
* **CoEvolve**: [A Joint Point Process Model for Information Diffusion and Network Co-evolution](https://arxiv.org/abs/1507.02293) [JMLR-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/Networks-Learning/Coevolution?style=social)](https://github.com/Networks-Learning/Coevolution)






## Hawkes process
* [Efficient Non-parametric Bayesian Hawkes Processes](https://arxiv.org/abs/1810.03730) [IJCAI-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/RuiZhang2016/Efficient-Nonparametric-Bayesian-Hawkes-Processes?style=social)](https://github.com/RuiZhang2016/Efficient-Nonparametric-Bayesian-Hawkes-Processes)
* [Bayesian Nonparametric Hawkes Processes](https://is.mpg.de/uploads_file/attachment/attachment/462/BNPNIPS2018_paper_14.pdf) [2018] 
* [Nonlinear Hawkes Processes in Time-Varying System](https://arxiv.org/abs/2106.04844) [arXiv-2021]

* **ProActive**: [Self-Attentive Temporal Point Process Flows for Activity Sequences](https://arxiv.org/abs/2206.05291) [KDD-2022] 

## Adversarial
* [Improving Maximum Likelihood Estimation of Temporal Point Process via Discriminative and Adversarial Learning](https://www.ijcai.org/Proceedings/2018/0409.pdf) [IJCAI-2018] 
* [Adversarial Training Model Unifying Feature Driven and Point Process Perspectives for Event Popularity Prediction](https://dl.acm.org/doi/abs/10.1145/3269206.3271714) [CIKM-2018] 

## Retrieval
* **NEUROSEQRET**: [Learning Temporal Point Processes for Efficient Retrieval of Continuous Time Event Sequences](https://arxiv.org/abs/2202.11485) [AAAI-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/data-iitd/neuroseqret?style=social)](https://github.com/data-iitd/neuroseqret)



## Others
* [Neural Point Process for Learning Spatiotemporal Event Dynamics](https://arxiv.org/abs/2112.06351) [L4DC-2022] [![GitHub Repo stars](https://img.shields.io/github/stars/Rose-STL-Lab/DeepSTPP?style=social)](https://github.com/Rose-STL-Lab/DeepSTPP)
* [Deep Structural Point Process for Learning Temporal Interaction Networks](https://arxiv.org/abs/2107.03573v1) [ECML PKDD-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/cjx96/DSPP?style=social)](https://github.com/cjx96/DSPP)
* [Learning to Select Exogenous Events for Marked Temporal Point Process](https://proceedings.neurips.cc/paper/2021/hash/032abcd424b4312e7087f434ef1c0094-Abstract.html) [NeurIPS-2021]
* **NEST**: [Imitation Learning of Neural Spatio-Temporal Point Processes](https://arxiv.org/pdf/1906.05467.pdf) [arXiv-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes?style=social)](https://github.com/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes)
* **UNIPoint**: [Universally Approximating Point Processes Intensities](https://arxiv.org/abs/2007.14082) [AAAI-2021]
* **NDTT**: [Neural Datalog through time: Informed temporal modeling via logical specification](https://arxiv.org/abs/2006.16723) [ICML-2020][![GitHub Repo stars](https://img.shields.io/github/stars/hongyuanmei/neural-datalog-through-time?style=social)](https://github.com/hongyuanmei/neural-datalog-through-time)
* [Deep Mixture Point Processes: Spatio-temporal Event Prediction with Rich Contextual Information](https://arxiv.org/abs/1906.08952) [KDD-2019]



# Workshop papers
* [Semi-supervised Learning for Marked Temporal Point Processes](https://arxiv.org/abs/2107.07729?context=cs) [IJCAI Workshop-2021]
* [Intermittent Demand Forecasting with Deep Renewal Processes](https://arxiv.org/abs/1911.10416) [Workshop on TPPs at NeurIPS-2019]


# Workshops
* [Learning with TPPs](https://sites.google.com/view/tpp-neurips-2019) [NeurIPS-2019]

# Applications
* Recommendation
  * **JODIE**: [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://arxiv.org/abs/1908.01207) [KDD-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/srijankr/jodie?style=social)](https://github.com/srijankr/jodie/)
  * [Time is of the Essence: a Joint Hierarchical RNN and Point Process Model for Time and Item Predictions](https://arxiv.org/abs/1812.01276) [WSDM-2019] [![GitHub Repo stars](https://img.shields.io/github/stars/BjornarVass/Recsys?style=social)](https://github.com/BjornarVass/Recsys)
* Human mobility and activity
  * [Recurrent spatio-temporal point process for check-in time prediction](https://dl.acm.org/doi/pdf/10.1145/3269206.3272003) [CIKM-2018] 
  * **Deepmove**: [Predicting human mobility with attentional recurrent networks](https://dl.acm.org/doi/10.1145/3178876.3186058) [WWW-2018] [![GitHub Repo stars](https://img.shields.io/github/stars/vonfeng/DeepMove?style=social)](https://github.com/vonfeng/DeepMove)
  * [Egocentric activity prediction via event modulated attention](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Shen_Egocentric_Activity_Prediction_ECCV_2018_paper.pdf) [ECCV-2018] 
* Event clustering
  * [Modeling the Dynamics of Learning Activity on the Web](https://dl.acm.org/doi/10.1145/3038912.3052669) [WWW-2017] 
  * [A Dirichlet mixture model of Hawkes processes for event sequence clustering](https://arxiv.org/abs/1701.09177) [NeurIPS-2017] [![GitHub Repo stars](https://img.shields.io/github/stars/HongtengXu/Hawkes-Process-Toolkit?style=social)](https://github.com/HongtengXu/Hawkes-Process-Toolkit)
* Anomaly detection
  * [Detecting Anomalous Event Sequences with Temporal Point Processes](https://papers.neurips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html) [NeurIPS-2021] [![GitHub Repo stars](https://img.shields.io/github/stars/shchur/tpp-anomaly-detection?style=social)](https://github.com/shchur/tpp-anomaly-detection)
  * [Detecting Changes in Dynamic Events Over Networks](https://ieeexplore.ieee.org/document/7907333) [IEEE Transactions on Signal and Information Processing over Networks-2017]
* Optimal control
  * [Steering social activity: A stochastic optimal control point of view](https://arxiv.org/abs/1802.07244) [JMLR-2018] 
  * [Enhancing human learning via spaced repetition optimization](https://www.pnas.org/doi/10.1073/pnas.1815156116) [PNAS-2019] 
* Misinformation on social media
  * **VigDet**: [Knowledge Informed Neural Temporal Point Process for Coordination Detection on Social Media](https://arxiv.org/abs/2110.15454v1) [NeurIPS-2021]
  * [Leveraging the crowd to detect and reduce the spread of fake news and misinformation](https://dl.acm.org/doi/10.1145/3159652.3159734) [WSDM-2018] 
* [Recurrent Poisson process unit for speech recognition](https://ojs.aaai.org/index.php/AAAI/article/view/4620) [AAAI-2019]
* [Point process latent variable models of larval zebrafish behavior](https://proceedings.neurips.cc/paper/2018/hash/e02af5824e1eb6ad58d6bc03ac9e827f-Abstract.html) [NeurIPS-2018]

# Books
* An introduction to the theory of point processes: Volume I: Elementary theory and methods: Daley, D.J., Vere-Jones, D
* An introduction to the theory of point processes: Volume II: General theory and structure: Daley, D.J., Vere-Jones, D


# Other awesome-tpp repos
* [awesome-ml4tpp](https://github.com/Thinklab-SJTU/awesome-ml4tpp)
* [Awesome-Temporal-Point-Process](https://github.com/aachenhang/Awesome-Temporal-Point-Process)
