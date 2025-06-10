**DAO-GP Drift Aware Online Non-Linear Regression Gaussian-Process**

**DAO-GP Architecture:**
The DAO-GP framework is engineered for superior performance in non-stationary data streams, seamlessly integrating four critical components to ensure robust adaptability. At its core, the **Online Gaussian Process (GP) Regression Engine** continuously updates its predictive function with each new data instance, with adaptation intrinsically tied to real-time concept drift assessment. This is complemented by a **Memory-Based Online Drift Detection and Magnitude Quantification** mechanism, which monitors and records performance indicators (KPIs) of incoming data, proactively identifying concept drift and categorizing its severity. A sophisticated **Hyperparameters Optimization** component dynamically refines the GP’s predictive accuracy by optimizing hyperparameters using the Negative Log Marginal Likelihood (NLML). Finally, a **Dynamic Kernel Pool for Adaptive Selection** enables DAO-GP to select the most appropriate kernel from a predefined set, ensuring responsiveness and robust adaptation to even the most severe distributional shifts. The synergistic interplay of these components is illustrated in Model-Flow.png.

<img src="https://github.com/anonymous273800/DAO-GP/blob/08013799c4377b01c0bbe3f4a838c57ccf7e77c9/Model-Flow.png" alt="image alt" width="50%" height="auto">


**Run Instructions:** 
1. No external libraries are needed, the only libraries needed are the basic python libraries (e.g. numpy, matplotlib, etc.)
2. The project contains 5 types of experiments all located under Experiments > 001PaperFinalExperiments >  
   A. 001StationaryNonLinearRegression-2DVisualization  
   B. 002StationaryNonLinearRegressionStressTest  
   C. 003DriftNonLinearRegression-2DVisualization  
   D. 004DriftNonLinearRegressionRobustness  
   E. 005BenchmarkWithOtherModels  


**DAO-GP Contributions:**
1. Introducing DAO-GP, a novel online regression model designed for robust and adaptive learning in dynamic environments.
2. DAO-GP incorporates a built-in drift detection mechanism that not only identifies distributional shifts but also classifies them by magnitude, enabling targeted adaptation strategies.
3.  DAO-GP triggers kernel hyperparameter optimization only upon drift detection, thereby avoiding the inefficiencies of per-batch tuning.
4.  The model maintains sparsity through inducing points and integrates a decay mechanism to down-weight outdated data, ensuring responsiveness to recent trends while reducing memory overhead.
5.  DAO-GP is inherently hyperparameter-free, an essential property for online learning, where continuous data flow makes manual tuning infeasible.
6.  Furthermore, it employs a diverse kernel pool to adaptively respond to distributional changes.
7.  Computational efficiency is achieved via the Woodbury Matrix Identity, which reduces the complexity of kernel updates from O(n^3) to O(m^2n), where m ≪ n, enabling scalable and memory-efficient learning in streaming settings.  



