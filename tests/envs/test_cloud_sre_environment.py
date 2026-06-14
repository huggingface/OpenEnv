"""
Tests for the Cloud SRE Environment.

Validates all three tasks (easy, medium, hard) with both fixed
and seeded states, plus grading correctness.
"""

import pytest
from cloud_sre_env.server.cloud_sre_environment import (
    CloudSREEnvironment, TASK_REGISTRY,
)
from cloud_sre_env.models import SREAction, ActionCommand


class TestEnvironmentBasics:
    """Test reset, step, state lifecycle."""

    def test_reset_returns_observation(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        assert len(obs.resources) > 0
        assert obs.step_number == 0
        assert obs.task_description != ""

    def test_step_increments(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        obs = env.step(SREAction(command="wait"))
        assert obs.step_number == 1

    def test_state_returns_sre_state(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        state = env.state
        assert state.task_id == "phantom_volume_cleanup"
        assert state.episode_id != ""

    def test_max_steps_ends_episode(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for _ in range(15):
            env.step(SREAction(command="wait"))
        assert env.state.done is True

    def test_invalid_task_raises(self):
        env = CloudSREEnvironment()
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset(task_id="nonexistent_task")


class TestTask1PhantomVolumeCleanup:
    """Test the easy task — orphan volume termination."""

    def test_fixed_state_has_orphans(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="phantom_volume_cleanup")
        orphans = [r for r in obs.resources if r.get("status") == "available"]
        assert len(orphans) == 3

    def test_perfect_score(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        for oid in ["ebs-orphan-001", "ebs-orphan-002", "ebs-orphan-003"]:
            env.step(SREAction(command="terminate", resource_id=oid))
        score, breakdown = env.grade()
        assert score == 1.0
        assert len(breakdown["orphans_terminated"]) == 3

    def test_penalty_for_wrong_termination(self):
        env = CloudSREEnvironment()
        env.reset(task_id="phantom_volume_cleanup")
        env.step(SREAction(command="terminate", resource_id="ec2-web-001"))
        score, breakdown = env.grade()
        assert score == 0.0
        assert len(breakdown["active_resources_terminated"]) == 1

    def test_seeded_state_is_different(self):
        env = CloudSREEnvironment()
        obs1 = env.reset(task_id="phantom_volume_cleanup", seed=42)
        obs2 = env.reset(task_id="phantom_volume_cleanup", seed=99)
        ids1 = {r["id"] for r in obs1.resources}
        ids2 = {r["id"] for r in obs2.resources}
        assert ids1 != ids2


class TestTask2LatencySpikeRemediation:
    """Test the medium task — RDS scaling."""

    def test_fixed_state_has_rds(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="latency_spike_remediation")
        rds = [r for r in obs.resources if r.get("type") == "rds_database"]
        assert len(rds) >= 1

    def test_scaling_rds_scores(self):
        env = CloudSREEnvironment()
        env.reset(task_id="latency_spike_remediation")
        env.step(SREAction(command="scale", resource_id="rds-primary-001",
                           params={"target_size": "db.t3.medium"}))
        score, breakdown = env.grade()
        assert score >= 0.7
        assert breakdown["rds_scaled"] is True


class TestTask3NoisyNeighborIncident:
    """Test the hard task — rogue instance investigation."""

    def test_fixed_state_has_rogue_and_stopped_backend(self):
        env = CloudSREEnvironment()
        obs = env.reset(task_id="noisy_neighbor_incident")
        ids = {r["id"] for r in obs.resources}
        assert "ec2-rogue-test-001" in ids
        assert "ec2-backend-prod-001" in ids

    def test_perfect_incident_response(self):
        env = CloudSREEnvironment()
        env.reset(task_id="noisy_neighbor_incident")
        env.step(SREAction(command="inspect", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="terminate", resource_id="ec2-rogue-test-001"))
        env.step(SREAction(command="reboot", resource_id="ec2-backend-prod-001"))
        score, breakdown = env.grade()
        assert score == 1.0
        assert breakdown["inspected_rogue"] is True
        assert breakdown["terminated_rogue"] is True
        assert breakdown["rebooted_backend"] is True
        assert breakdown["alerts_resolved"] is True


class TestAllTasksRegistered:
    """Ensure all tasks are in the registry."""

    def test_registry_has_all_tasks(self):
        assert "phantom_volume_cleanup" in TASK_REGISTRY
        assert "latency_spike_remediation" in TASK_REGISTRY
        assert "noisy_neighbor_incident" in TASK_REGISTRY
