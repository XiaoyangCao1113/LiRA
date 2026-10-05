"""MABIM inventory management (ReplenishmentEnv): N=400 SKU agents, K=2 shared costs.

Modules
-------
``env``           all-agent ReplenishmentEnv adapter (rewards/costs scaled by 1e-6)
``learner``       shared FC-GRU categorical actor, centralized critic, checkpoints
``live_dual``     dimensionless ``C_k/d_k <= 1`` dual primitives
``transaction``   functional q-step PPO/Adam learner and DU + score-correction terms
``meta_gradient`` leave-one-out score-corrected LiRA outer gradient
``trainer``       Uniform / PAL / LiRA live-dual training loop
``evaluation``    held-out test-split rollouts with common random numbers

Importing this package does not import ReplenishmentEnv; the environment is
loaded lazily when a runner is built.
"""
