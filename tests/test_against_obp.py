#!/usr/bin/env python3
"""Check our estimators against obp's reference implementations on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.ope import (
    DirectMethod,
    DoublyRobust,
    InverseProbabilityWeighting,
    SelfNormalizedDoublyRobust,
    SelfNormalizedInverseProbabilityWeighting,
    SwitchDoublyRobust,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import ope  # noqa: E402


def main() -> None:
    data = SyntheticBanditDataset(n_actions=10, dim_context=5, reward_function=logistic_reward_function,
                                  behavior_policy_function=None, random_state=0)
    fb = data.obtain_batch_bandit_feedback(n_rounds=20_000)
    rng = np.random.default_rng(1)
    pi_e = rng.dirichlet(np.ones(10), size=fb["n_rounds"])
    q_hat = np.clip(fb["expected_reward"] + rng.normal(0, 0.05, fb["expected_reward"].shape), 0, 1)
    common = dict(pi_e=pi_e, action=fb["action"], reward=fb["reward"], pscore=fb["pscore"], q_hat=q_hat)
    obp_kwargs = dict(reward=fb["reward"], action=fb["action"], pscore=fb["pscore"],
                      action_dist=pi_e[:, :, None], estimated_rewards_by_reg_model=q_hat[:, :, None],
                      position=np.zeros(fb["n_rounds"], dtype=int))
    reference = {
        "dm": DirectMethod(),
        "ips": InverseProbabilityWeighting(),
        "snips": SelfNormalizedInverseProbabilityWeighting(),
        "dr": DoublyRobust(),
        "sndr": SelfNormalizedDoublyRobust(),
        "switch_dr": SwitchDoublyRobust(tau=50.0),
    }
    failed = False
    for name, estimator in reference.items():
        ours = ope.estimate(name, **common).mean()
        theirs = estimator.estimate_policy_value(**obp_kwargs)
        ok = abs(ours - theirs) < 1e-9
        failed |= not ok
        print(f"{name:10s} ours={ours:.6f} obp={theirs:.6f} {'OK' if ok else 'MISMATCH'}")
    print(f"true value  {data.calc_ground_truth_policy_value(fb['expected_reward'], pi_e[:, :, None]):.6f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
