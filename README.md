**DAO-GP Drift Aware Online Non-Linear Regression Gaussian-Process**

**DAO-GP Architecture:**
The DAO-GP framework is engineered for superior performance in non-stationary data streams, seamlessly integrating four critical components to ensure robust adaptability. At its core, the **Online Gaussian Process (GP) Regression Engine** continuously updates its predictive function with each new data instance, with adaptation intrinsically tied to real-time concept drift assessment. This is complemented by a **Memory-Based Online Drift Detection and Magnitude Quantification** mechanism, which monitors and records performance indicators (KPIs) of incoming data, proactively identifying concept drift and categorizing its severity. A sophisticated **Hyperparameters Optimization** component dynamically refines the GP’s predictive accuracy by optimizing hyperparameters using the Negative Log Marginal Likelihood (NLML). Finally, a **Dynamic Kernel Pool for Adaptive Selection** enables DAO-GP to select the most appropriate kernel from a predefined set, ensuring responsiveness and robust adaptation to even the most severe distributional shifts. The synergistic interplay of these components is illustrated in Model-Flow.png.

<img src="https://github.com/anonymous273800/DAO-GP/blob/08013799c4377b01c0bbe3f4a838c57ccf7e77c9/Model-Flow.png" alt="image alt" width="50%" height="auto">

