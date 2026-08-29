# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fraud Triage Environment server-side logic.

Implements the classic reset() / step() / state OpenEnv Environment
interface for a sequential transaction-risk-decisioning task.
"""

from __future__ import annotations

import os
import uuid

from openenv.core.env_server import Environment

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import FraudTriageAction, FraudTriageObservation, FraudTriageState
    from .transaction_generator import TransactionGenerator
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import FraudTriageAction, FraudTriageObservation, FraudTriageState
    from server.transaction_generator import TransactionGenerator


# Reward structure. Asymmetric on purpose: missing real fraud (false
# negative) is penalized far more heavily than annoying a genuine customer
# (false positive), which mirrors how production risk teams actually weight
# these outcomes -- but flagging/escalating everything is also penalized so
# the agent can't "win" by being maximally conservative.
REWARD_TRUE_POSITIVE = 1.0  # correctly flagged/escalated/blocked fraud
REWARD_TRUE_NEGATIVE = 0.1  # correctly approved a legitimate transaction
REWARD_FALSE_POSITIVE = -0.5  # incorrectly flagged/escalated/blocked a legit txn
REWARD_FALSE_NEGATIVE = -3.0  # incorrectly approved a fraudulent txn

# Escalation costs more than a flag even when correct, since it consumes
# human analyst time -- so the agent learns to reserve it for genuinely
# ambiguous cases rather than defaulting to it.
ESCALATION_COST = -0.1


class FraudTriageEnvironment(Environment):
    """
    Sequential transaction risk-decisioning environment.

    Each episode streams `episode_length` synthetic transactions. For each
    transaction the agent chooses one of four actions (approve / flag /
    escalate / block); the environment scores the decision against the
    transaction's hidden ground-truth fraud label and returns the next
    transaction as the following observation.
    """

    def __init__(self, episode_length: int = 200, fraud_rate: float = 0.12):
        super().__init__()
        self.episode_length = episode_length
        self.fraud_rate = fraud_rate
        self.reset()

    def reset(self):
        seed = int.from_bytes(os.urandom(4), "little")
        self._generator = TransactionGenerator(fraud_rate=self.fraud_rate, seed=seed)
        self._current_txn = None

        self._state = FraudTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            episode_length=self.episode_length,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            true_negatives=0,
            cumulative_reward=0.0,
        )
        return self._make_observation(advance=True)

    def step(self, action: FraudTriageAction):
        if self._current_txn is None:
            # Defensive: step() called before a transaction was ever issued.
            return self._make_observation(advance=True)

        txn = self._current_txn
        decision = action.decision
        is_flagging_action = decision in (1, 2, 3)  # FLAG, ESCALATE, BLOCK

        if txn.is_fraud and is_flagging_action:
            reward = REWARD_TRUE_POSITIVE
            self._state.true_positives += 1
        elif txn.is_fraud and not is_flagging_action:
            reward = REWARD_FALSE_NEGATIVE
            self._state.false_negatives += 1
        elif not txn.is_fraud and is_flagging_action:
            reward = REWARD_FALSE_POSITIVE
            self._state.false_positives += 1
        else:
            reward = REWARD_TRUE_NEGATIVE
            self._state.true_negatives += 1

        if decision == 2:  # ESCALATE
            reward += ESCALATION_COST

        self._state.cumulative_reward += reward
        self._state.step_count += 1

        done = self._state.step_count >= self.episode_length
        return self._make_observation(advance=not done, reward=reward, done=done)

    def _make_observation(self, advance: bool, reward: float = 0.0, done: bool = False):
        if advance and not done:
            self._current_txn = self._generator.sample()

        txn = self._current_txn
        return FraudTriageObservation(
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            merchant_category=txn.merchant_category,
            velocity_1h=txn.velocity_1h,
            velocity_24h=txn.velocity_24h,
            amount_zscore=txn.amount_zscore,
            is_new_merchant=txn.is_new_merchant,
            is_new_device=txn.is_new_device,
            cross_border=txn.cross_border,
            hour_of_day=txn.hour_of_day,
            account_age_days=txn.account_age_days,
            reward=reward,
            done=done,
            metadata={
                "step_count": self._state.step_count,
                "cumulative_reward": self._state.cumulative_reward,
            },
        )

    @property
    def state(self):
        return self._state
