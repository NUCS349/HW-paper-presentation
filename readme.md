

Paper List that you can directly draw from:

1. Nonlinear dimensionality reduction by locally linear embedding. By Roweis, Sam T., and Lawrence K. Saul. science 290.5500 (2000): 2323-2326.

Summary: Locally linear embedding (LLE) seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods. It can be thought of as a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding.

2. Auto-encoding variational bayes by Diederik P Kingma, Max Welling

Summary: How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. 

3. Dropout: a simple way to prevent neural networks from overfitting, by Hinton, G.E., Krizhevsky, A., Srivastava, N., Sutskever, I., & Salakhutdinov, R. (2014). Journal of Machine Learning Research, 15, 1929-1958. (cited 2084 times, HIC: 142 , CV: 536).

Summary: The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much. This significantly reduces overfitting and gives major improvements over other regularization methods

4. Deep Residual Learning for Image Recognition, by He, K., Ren, S., Sun, J., & Zhang, X. (2016). CoRR, abs/1512.03385. (cited 1436 times, HIC: 137 , CV: 582).

Summary: We present a residual learning framework to ease the training of deep neural networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.

5. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, by Sergey Ioffe, Christian Szegedy (2015) ICML. (cited 946 times, HIC: 56 , CV: 0).

Summary: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change.  We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs.  Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin.

6. Microsoft COCO: Common Objects in Context , by Belongie, S.J., Dollár, P., Hays, J., Lin, T., Maire, M., Perona, P., Ramanan, D., & Zitnick, C.L. (2014). ECCV. (cited 830 times, HIC: 78 , CV: 279) Summary: We present a new dataset with the goal of advancing the state-of-the-art in object recognition by placing the question of object recognition in the context of the broader question of scene understanding. Our dataset contains photos of 91 objects types that would be easily recognizable by a 4 year old. Finally, we provide baseline performance analysis for bounding box and segmentation detection results using a Deformable Parts Model.

7. Learning deep features for scene recognition using places database , by Lapedriza, À., Oliva, A., Torralba, A., Xiao, J., & Zhou, B. (2014). NIPS. (cited 644 times, HIC: 65 , CV: 0)

Summary: We introduce a new scene-centric database called Places with over 7 million labeled pictures of scenes. We propose new methods to compare the density and diversity of image datasets and show that Places is as dense as other scene datasets and has more diversity.

8. High-Speed Tracking with Kernelized Correlation Filters, by Batista, J., Caseiro, R., Henriques, J.F., & Martins, P. (2015). CoRR, abs/1404.7584. (cited 439 times, HIC: 43 , CV: 0)

Summary: In most modern trackers,  to cope with natural image changes, a classifier is typically trained with translated and scaled sample patches. We propose an analytic model for datasets of thousands of translated patches. By showing that the resulting data matrix is circulant, we can diagonalize it with the discrete Fourier transform, reducing both storage and computation by several orders of magnitude.

9. Review on Multi-Label Learning Algorithms, by  Zhang, M., & Zhou, Z. (2014). IEEE TKDE,  (cited 436 times, HIC: 7 , CV: 91)

Summary:  This paper aims to provide a timely review on multi-label learning studies the problem where each example is represented by a single instance while associated with a set of labels simultaneously.

10. How transferable are features in deep neural networks, by Bengio, Y., Clune, J., Lipson, H., & Yosinski, J. (2014) CoRR, abs/1411.1792. (cited 402 times, HIC: 14 , CV: 0)

Summary: We experimentally quantify the generality versus specificity of neurons in each layer of a deep convolutional neural network and report a few surprising results. Transferability is negatively affected by two distinct issues: (1) the specialization of higher layer neurons to their original task at the expense of performance on the target task, which was expected, and (2) optimization difficulties related to splitting networks between co-adapted neurons, which was not expected.

11. Do we need hundreds of classifiers to solve real world classification problems, by Amorim, D.G., Barro, S., Cernadas, E., & Delgado, M.F. (2014).  Journal of Machine Learning Research (cited 387 times, HIC: 3 , CV: 0)

Summary: We evaluate 179 classifiers arising from 17 families (discriminant analysis, Bayesian, neural networks, support vector machines, decision trees, rule-based classifiers, boosting, bagging, stacking, random forests and other ensembles, generalized linear models, nearest-neighbors, partial least squares and principal component regression, logistic and multinomial regression, multiple adaptive regression splines and other methods). We use 121 data sets from UCI data base to study the classifier behavior, not dependent on the data set collection. The winners are the random forest (RF) versions implemented in R and accessed via caret) and the SVM with Gaussian kernel implemented in C using LibSVM.


12. Knowledge vault: a web-scale approach to probabilistic knowledge fusion, by Dong, X., Gabrilovich, E., Heitz, G., Horn, W., Lao, N., Murphy, K., ... & Zhang, W. (2014, August). In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining  ACM. (cited 334 times, HIC: 7 , CV: 107).

Summary: We introduce Knowledge Vault, a Web-scale probabilistic knowledge base that combines extractions from Web content (obtained via analysis of text, tabular data, page structure, and human annotations) with prior knowledge derived from existing knowledge repositories for constructing knowledge bases. We employ supervised machine learning methods for fusing  distinct information sources. The Knowledge Vault is substantially bigger than any previously published structured knowledge repository, and features a probabilistic inference system that computes calibrated probabilities of fact correctness.

13. Scalable Nearest Neighbor Algorithms for High Dimensional Data, by Lowe, D.G., & Muja, M. (2014). IEEE Trans. Pattern Anal. Mach. Intell., (cited 324 times, HIC: 11 , CV: 69).

Summary: We propose new algorithms for approximate nearest neighbor matching and evaluate and compare them with previous algorithms.  In order to scale to very large data sets that would otherwise not fit in the memory of a single machine, we propose a distributed nearest neighbor matching framework that can be used with any of the algorithms described in the paper.

14. Playing atari with deep reinforcement learning by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller

Summary: We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.


15. Trends in extreme learning machines: a review, by Huang, G., Huang, G., Song, S., & You, K. (2015).  Neural Networks,  (cited 323 times, HIC: 0 , CV: 0)

Summary: We aim to report the current state of the theoretical research and practical advances on Extreme learning machine (ELM). Apart from classification and regression, ELM has recently been extended for clustering, feature selection, representational learning and many other learning tasks.  Due to its remarkable efficiency, simplicity, and impressive generalization performance, ELM have been applied in a variety of domains, such as biomedical engineering, computer vision, system identification, and control and robotics.


16. A survey on concept drift adaptation, by Bifet, A., Bouchachia, A., Gama, J., Pechenizkiy, M., & Zliobaite, I.  ACM Comput. Surv., 2014 , (cited 314 times, HIC: 4 , CV: 23)

Summary: This work aims at providing a comprehensive introduction to the concept drift adaptation that refers to an online supervised learning scenario when the relation between the input data and the target variable changes over time.


17. Simultaneous Detection and Segmentation, by Arbeláez, P.A., Girshick, R.B., Hariharan, B., & Malik, J. (2014) ECCV , (cited 286 times, HIC: 23 , CV: 94)

Summary: We aim to detect all instances of a category in an image and, for each instance, mark the pixels that belong to it. We call this task Simultaneous Detection and Segmentation (SDS).

18. A survey on feature selection methods, by Chandrashekar, G., & Sahin, F.  Int. J. on Computers & Electrical Engineering, (cited 279 times, HIC: 1 , CV: 58)

Summary: Plenty of feature selection methods are available in literature due to the availability of data with hundreds of variables leading to data with very high dimension.

19. One Millisecond Face Alignment with an Ensemble of Regression Trees, by Kazemi, Vahid, and Josephine Sullivan, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2014, (cited 277 times, HIC: 15 , CV: 0)

Summary: This paper addresses the problem of Face Alignment for a single image. We show how an ensemble of regression trees can be used to estimate the face's landmark positions directly from a sparse subset of pixel intensities, achieving super-realtime performance with high quality predictions.

20. A survey of multiple classifier systems as hybrid systems , by Corchado, E., Graña, M., & Wozniak, M. (2014). Information Fusion, 16, 3-17. (cited 269 times, HIC: 1 , CV: 22)

Summary: A current focus of intense research in pattern classification is the combination of several classifier systems, which can be built following either the same or different models and/or datasets building.

21. The pagerank citation ranking: Bringing order to the web. By L Page, S Brin, R Motwani, T Winograd - 1999 - ilpubs.stanford.edu

Summary: The importance of a Web page is an inherently subjective matter, which depends on the readers interests, knowledge and attitudes. But there is still much that can be said objectively about the relative importance of Web pages. 

22. Learning with known operators reduces maximum error bounds by Maier, A. K., Syben, C., Stimpel, B., Würfl, T., Hoffmann, M., Schebesch, F., ... & Christiansen, S. (2019)

Summary: We describe an approach for incorporating prior knowledge into machine learning algorithms. We aim at applications in physics and signal processing in which we know that certain operations must be embedded into the algorithm. Any operation that allows computation of a gradient or sub-gradient towards its inputs is suited for our framework. 

23. Visualizing data using t-SNE by Laurens van der Maaten, Geoffrey Hinton

Summary: We present a new technique called" t-SNE" that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. 

24. Unsupervised representation learning with deep convolutional generative adversarial networks by Alec Radford, Luke Metz, Soumith Chintala

Summary: In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning

25. The tree ensemble layer: Differentiability meets conditional computation (Hazimeh, H., Ponomareva, N., Mol, P., Tan, Z., & Mazumder, R. (2020, November).)

Summary: Neural networks and tree ensembles are state-of-the-art learners, each with its unique
statistical and computational advantages. We aim to combine these advantages by
introducing a new layer for neural networks, composed of an ensemble of differentiable
decision trees (aka soft trees). While differentiable trees demonstrate promising results in the
literature, they are typically slow in training and inference as they do not support conditional
computation.

26. Yolov4: Optimal speed and accuracy of object detection ( A Bochkovskiy, CY Wang, HYM Liao )

Summary: There are a huge number of features which are said to improve Convolutional Neural
Network (CNN) accuracy. Practical testing of combinations of such features on large
datasets, and theoretical justification of the result, is required. Some features operate on
certain models exclusively and for certain problems exclusively, or only for small-scale
datasets; while some features, such as batch-normalization and residual-connections, are
applicable to the majority of models, tasks, and datasets.

27. ResNeSt: Split-Attention Networks ( H Zhang, C Wu, Z Zhang, Y Zhu, H Lin, Z Zhang )

Summary: It is well known that featuremap attention and multi-path representation are important for visual recognition. In this paper, we present a modularized architecture, which applies the channel-wise attention on different network branches to leverage their success in capturing cross-feature interactions and learning diverse representations. Our design results in a simple and unified computation block, which can be parameterized using only a few variables. Our model, named ResNeSt, outperforms EfficientNet in accuracy and latency trade-off on image classification.

28. Training with Quantization Noise for Extreme Model Compression  (  Fan, Angela; Stock, Pierre; Graham, Benjamin; Grave, Edouard; Gribonval, Remi; Jegou, Herve; Joulin, Armand )

We tackle the problem of producing compact models, maximizing their accuracy for a given model size. A standard solution is to train networks with Quantization Aware Training, where the weights are quantized during training and the gradients approximated with the Straight-Through Estimator. 

29. End-to-end object detection with transformers ( N Carion, F Massa, G Synnaeve, N Usunier )

Summary: We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. 

30. Language models are few-shot learners (TB Brown, B Mann, N Ryder, M Subbiah)

Summary: Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by
pre-training on a large corpus of text followed by fine-tuning on a specific task. While
typically task-agnostic in architecture, this method still requires task-specific fine-tuning
datasets of thousands or tens of thousands of examples. By contrast, humans can generally
perform a new language task from only a few examples or from simple instructions-
something which current NLP systems still largely struggle to do. 

31. Unsupervised translation of programming languages (MA Lachaux, B Roziere, L Chanussot)

Summary: A transcompiler, also known as source-to-source translator, is a system that converts source
code from a high-level programming language (such as C++ or Python) to another.
Transcompilers are primarily used for interoperability, and to port codebases written in an
obsolete or deprecated language (eg COBOL, Python 2) to a modern one. 

32. DeepFaceDrawing: Deep Generation of Face Images from Sketches ( SY Chen, W Su, L Gao, S Xia, H Fu  )

Summary: Recent deep image-to-image translation techniques allow fast generation of face images
from freehand sketches. However, existing solutions tend to overfit to sketches, thus
requiring professional sketches or even edge maps as input. To address this issue, our key
idea is to implicitly model the shape space of plausible face images and synthesize a face
image in this space to approximate an input sketch. We take a local-to-global approach. 

33. Learning to Match Distributions for Domain Adaptation ( Chaohui Yu, Jindong Wang, Chang Liu, Tao Qin, Renjun Xu, Wenjie Feng, Yiqiang Chen, Tie-Yan Liu )

Summary: When the training and test data are from different distributions, domain adaptation is needed to reduce dataset bias to improve the model's generalization ability. Since it is difficult to directly match the cross-domain joint distributions, existing methods tend to reduce the marginal or conditional distribution divergence using predefined distances such as MMD and adversarial-based discrepancies. However, it remains challenging to determine which method is suitable for a given application since they are built with certain priors or bias. Thus they may fail to uncover the underlying relationship between transferable features and joint distributions

34. Tensorflow quantum: A software framework for quantum machine learning ( M Broughton, G Verdon, T McCourt, AJ Martinez ... )

Summary: We introduce TensorFlow Quantum (TFQ), an open source library for the rapid prototyping of
hybrid quantum-classical models for classical or quantum data. This framework offers high-
level abstractions for the design and training of both discriminative and generative quantum
models under TensorFlow and supports high-performance quantum circuit simulators. 


Here are some more links where you can draw papers from if you cannot find something interesting in the list above:

1. https://analyticsindiamag.com/best-machine-learning-papers-2019-nips-icml-ai/
2. https://www.topbots.com/top-ml-research-papers-2019/
3. https://www.topbots.com/ai-machine-learning-research-papers-2020/
4. https://www.linkedin.com/pulse/top-7-machine-learning-papers-2019-laxmi-kant-tiwari/
5. https://www.kdnuggets.com/2017/04/top-20-papers-machine-learning.html
6. https://towardsdatascience.com/10-overlooked-machine-learning-advances-in-the-last-10-decades-2e9fe9f2f073
