# SPDX-License-Identifier: BSD-3-Clause

"""Test Fraud Triage environment client and server integration."""

import os
import signal
import subprocess
import sys
import time
import unittest

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.fraud_triage_env import (
    FraudTriageAction,
    FraudTriageEnv,
    FraudTriageObservation,
    TriageDecision,
)


class TestFraudTriageEnv(unittest.TestCase):
    server_process = None
    client = None

    @classmethod
    def setUpClass(cls):
        cls.server_process = subprocess.Popen(
            ["python", "-m", "envs.fraud_triage_env.server.app"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)

        for _ in range(10):
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("Fraud Triage server did not start")

        cls.client = FraudTriageEnv(base_url="http://127.0.0.1:8000")

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(cls.server_process.pid, signal.SIGKILL)
            for stream in [
                cls.server_process.stdin,
                cls.server_process.stdout,
                cls.server_process.stderr,
            ]:
                if stream and not stream.closed:
                    stream.close()

    def test_reset_returns_valid_observation(self):
        result = self.client.reset()
        observation = result.observation

        assert isinstance(observation, FraudTriageObservation)
        assert observation.transaction_id != ""
        assert observation.amount > 0
        assert 0 <= observation.merchant_category <= 9
        assert 0 <= observation.hour_of_day <= 23
        assert observation.legal_actions == [0, 1, 2, 3]
        assert not observation.done
        assert observation.reward == 0.0

    def test_approve_and_block_actions_return_reward(self):
        self.client.reset()

        result = self.client.step(FraudTriageAction(decision=TriageDecision.APPROVE))
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

        result = self.client.step(FraudTriageAction(decision=TriageDecision.BLOCK))
        assert isinstance(result.reward, float)

    def test_episode_terminates_after_episode_length_steps(self):
        self.client.reset()
        done = False
        steps = 0
        max_steps = 500  # safety cap well above default episode_length=200

        while not done and steps < max_steps:
            result = self.client.step(FraudTriageAction(decision=TriageDecision.FLAG))
            done = result.done
            steps += 1

        assert done, "Episode should terminate within episode_length steps"
        assert steps <= max_steps

    def test_state_tracks_confusion_matrix_counts(self):
        self.client.reset()
        for _ in range(20):
            result = self.client.step(FraudTriageAction(decision=TriageDecision.APPROVE))
            if result.done:
                break

        state = self.client.state()
        total_outcomes = (
            state.true_positives
            + state.false_positives
            + state.false_negatives
            + state.true_negatives
        )
        assert total_outcomes > 0
        assert state.step_count > 0


if __name__ == "__main__":
    unittest.main()
