"""Melting Pot Commons Harvest (``commons_harvest__open``, N=7, K=1).

Requires a Melting Pot source checkout at revision
``817f8c1974863a91909c04c7a69dd33993199ec6`` with
``patches/meltingpot_apple_count.patch`` applied (it exposes the live apple
count as an observation without changing rewards or dynamics), plus
``dmlab2d`` and TensorFlow for the pretrained Harvest bot.

Modules:

* :mod:`.env` -- substrate wrapper and ecological-depletion cost.
* :mod:`.adapter` -- learning adapter (compact observations, bot action prior).
* :mod:`.learner` -- masked-categorical PPO learner (``uniform``/``pal`` arms).
* :mod:`.transaction` -- functional q-step lookahead (direct unroll + score term).
* :mod:`.meta_batch` -- leave-one-out score-corrected allocation gradient.
"""

from .env import N_ACTIONS, N_AGENTS, N_CONSTRAINTS, SOURCE_COMMIT, SUBSTRATE

__all__ = ["N_ACTIONS", "N_AGENTS", "N_CONSTRAINTS", "SOURCE_COMMIT", "SUBSTRATE"]
