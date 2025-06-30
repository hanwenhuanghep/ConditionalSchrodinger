SBCG: Schrodinger bridge based deep conditional generative learning

Conditional generative models mark a major advancement in machine learning by enabling controlled data generation through the incorporation of auxiliary information. Schrödinger bridge-based conditional generative learning (SBCG) builds on this framework by modeling a unit-time diffusion process governed by a stochastic differential equation (SDE), which transports a fixed point at time t=0t=0 to a target conditional distribution at t=1t=1. SBCG is implemented by discretizing the SDE using the Euler–Maruyama method and estimating the drift term nonparametrically with a deep neural network.

The file sbcg_simu.ipynb contains a Python implementation of the SBCG method applied to simulated data.
