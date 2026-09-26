# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for KubernetesProvider.

Provider-behavior tests inject a duck-typed fake adapter (``_adapter=``), so
they run without ``pip install kubernetes``. A separate test installs a minimal
fake ``kubernetes`` module to pin the real ``_DefaultKubernetesAdapter`` SDK
shape, including in-cluster config fallback and 404/409 handling that must not
leak API bodies.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime import kubernetes_provider
from openenv.core.containers.runtime.kubernetes_provider import (
    _DefaultKubernetesAdapter,
    KubernetesProvider,
)
from openenv.core.containers.runtime.providers import ContainerProvider


class _FakeAdapter:
    def __init__(self):
        self.pods: list[dict] = []
        self.services: list[dict] = []
        self.deleted_pods = 0
        self.deleted_services = 0
        self.ensured = 0
        self.phase = "Running"
        self.reason = None
        self.namespace_exists = True
        self.fail_pod: Exception | None = None
        self.fail_service: Exception | None = None
        self.fail_delete_pod: Exception | None = None

    def ensure_namespace(self):
        self.ensured += 1
        if not self.namespace_exists:
            raise RuntimeError(
                "Kubernetes namespace does not exist. "
                "KubernetesProvider does not create namespaces."
            )

    def create_pod(self, manifest):
        if self.fail_pod is not None:
            raise self.fail_pod
        self.pods.append(manifest)

    def create_service(self, manifest):
        if self.fail_service is not None:
            raise self.fail_service
        self.services.append(manifest)

    def delete_pod(self, name):
        self.deleted_pods += 1
        if self.fail_delete_pod is not None:
            raise self.fail_delete_pod

    def delete_service(self, name):
        self.deleted_services += 1

    def pod_status(self, name):
        return {"phase": self.phase, "reason": self.reason}


@pytest.fixture()
def adapter():
    return _FakeAdapter()


@pytest.fixture()
def provider(adapter):
    return KubernetesProvider(namespace="openenv", _adapter=adapter)


@pytest.fixture(autouse=True)
def _fast_provider_sleep():
    with patch("openenv.core.containers.runtime.kubernetes_provider.time.sleep"):
        yield


def _labels(manifest):
    return manifest["metadata"]["labels"]


class TestStartContainer:
    def test_returns_incluster_http_url(self, provider):
        url = provider.start_container("echo-env:latest")
        assert url.startswith("http://openenv-echo-env-")
        assert url.endswith(".openenv.svc.cluster.local:8000")
        assert not url.startswith("https://")
        assert provider.base_url == url

    def test_explicit_port_is_service_port_only(self, provider, adapter):
        url = provider.start_container("echo-env:latest", port=9000)
        assert url.endswith(".openenv.svc.cluster.local:9000")
        assert adapter.pods[0]["spec"]["containers"][0]["ports"] == [
            {"name": "http", "container_port": 8000}
        ]
        assert adapter.services[0]["spec"]["ports"][0]["target_port"] == 8000
        assert adapter.services[0]["spec"]["ports"][0]["port"] == 9000

    def test_image_name_is_dns_safe(self, provider, adapter):
        provider.start_container("registry.example.com/Org/Echo_Env:latest")
        name = adapter.pods[0]["metadata"]["name"]
        assert name.startswith("openenv-echo-env-")
        assert len(name) <= 63
        assert "_" not in name
        assert name == adapter.services[0]["metadata"]["name"]

    def test_long_image_name_stays_within_63_chars(self, provider, adapter):
        provider.start_container(("a" * 80) + ":latest")
        assert len(adapter.pods[0]["metadata"]["name"]) <= 63

    def test_unusable_image_name_raises_before_create(self, provider, adapter):
        with pytest.raises(ValueError, match="valid Kubernetes name"):
            provider.start_container("---:latest")
        assert adapter.pods == []
        assert adapter.ensured == 0

    def test_constructor_image_used_when_start_omits_image(self, adapter):
        provider = KubernetesProvider(
            _adapter=adapter, image="constructor-env:latest", namespace="openenv"
        )
        provider.start_container()
        assert adapter.pods[0]["spec"]["containers"][0]["image"] == (
            "constructor-env:latest"
        )

    def test_start_image_overrides_constructor_image(self, adapter):
        provider = KubernetesProvider(
            _adapter=adapter, image="constructor-env:latest", namespace="openenv"
        )
        provider.start_container("explicit-env:latest")
        assert adapter.pods[0]["spec"]["containers"][0]["image"] == (
            "explicit-env:latest"
        )

    def test_requires_image_from_constructor_or_start(self, adapter):
        provider = KubernetesProvider(_adapter=adapter)
        with pytest.raises(ValueError, match="requires an image"):
            provider.start_container()

    def test_env_vars_forwarded(self, provider, adapter):
        provider.start_container("echo-env:latest", env_vars={"TOKEN": "supersecret"})
        assert adapter.pods[0]["spec"]["containers"][0]["env"] == [
            {"name": "TOKEN", "value": "supersecret"}
        ]

    def test_empty_env_vars_override_constructor(self, adapter):
        provider = KubernetesProvider(
            _adapter=adapter,
            namespace="openenv",
            image="echo-env:latest",
            env_vars={"TOKEN": "supersecret"},
        )
        provider.start_container(env_vars={})
        assert "env" not in adapter.pods[0]["spec"]["containers"][0]

    def test_ownership_labels_override_caller(self, provider, adapter):
        provider.start_container(
            "echo-env:latest",
            labels={
                "app.kubernetes.io/managed-by": "attacker",
                "app.kubernetes.io/instance": "hijack",
                "team": "rl",
            },
        )
        labels = _labels(adapter.pods[0])
        name = adapter.pods[0]["metadata"]["name"]
        assert labels["app.kubernetes.io/managed-by"] == "openenv"
        assert labels["app.kubernetes.io/instance"] == name
        assert labels["app.kubernetes.io/part-of"] == "openenv"
        assert labels["team"] == "rl"
        assert adapter.services[0]["spec"]["selector"] == {
            "app.kubernetes.io/instance": name
        }
        assert _labels(adapter.services[0])["app.kubernetes.io/managed-by"] == (
            "openenv"
        )

    def test_invalid_label_value_rejected_before_create(self, provider, adapter):
        with pytest.raises(ValueError, match="label values"):
            provider.start_container(
                "echo-env:latest", labels={"team": "line\nsecret-value"}
            )
        assert adapter.pods == []
        assert "secret-value" not in repr(adapter.pods)

    def test_functional_kwargs(self, provider, adapter):
        provider.start_container(
            "echo-env:latest",
            resources={"requests": {"cpu": "100m"}, "limits": {"memory": "128Mi"}},
            runtime_class_name="gvisor",
            image_pull_secrets=["reg-cred"],
            service_account_name="openenv-runner",
        )
        spec = adapter.pods[0]["spec"]
        container = spec["containers"][0]
        assert container["resources"]["requests"] == {"cpu": "100m"}
        assert container["resources"]["limits"] == {"memory": "128Mi"}
        assert spec["runtime_class_name"] == "gvisor"
        assert spec["image_pull_secrets"] == ["reg-cred"]
        assert spec["service_account_name"] == "openenv-runner"
        assert spec["automount_service_account_token"] is False
        assert spec["restart_policy"] == "Never"

    def test_unknown_kwarg_rejected_before_create(self, provider, adapter):
        with pytest.raises(ValueError, match="ingress"):
            provider.start_container("echo-env:latest", ingress="yes")
        assert adapter.pods == []

    def test_invalid_namespace_rejected_before_api(self, adapter):
        with pytest.raises(ValueError, match="namespace"):
            KubernetesProvider(namespace="Not_A_Namespace", _adapter=adapter)
        assert adapter.ensured == 0

    def test_missing_namespace_raises_and_creates_nothing(self, provider, adapter):
        adapter.namespace_exists = False
        with pytest.raises(RuntimeError, match="does not exist"):
            provider.start_container("echo-env:latest")
        assert adapter.pods == []
        assert adapter.deleted_pods == 0

    def test_rejects_bad_port(self, provider):
        with pytest.raises(ValueError, match="port"):
            provider.start_container("echo-env:latest", port=0)
        with pytest.raises(ValueError, match="port"):
            provider.start_container("echo-env:latest", port=True)


class TestLifecycle:
    def test_double_start_raises_and_preserves_existing_pod(self, provider, adapter):
        provider.start_container("echo-env:latest")
        with pytest.raises(RuntimeError, match="already has an active Pod"):
            provider.start_container("other:latest")
        assert len(adapter.pods) == 1
        assert adapter.deleted_pods == 0

    def test_start_after_stop_is_allowed(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.stop_container()
        provider.start_container("other:latest")
        assert len(adapter.pods) == 2

    def test_stop_deletes_pod_and_service(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.stop_container()
        assert adapter.deleted_pods == 1
        assert adapter.deleted_services == 1
        assert provider._name is None
        assert provider._base_url is None

    def test_stop_without_start_is_noop(self, provider, adapter):
        provider.stop_container()
        assert adapter.deleted_pods == 0
        assert adapter.deleted_services == 0

    def test_stop_is_idempotent(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.stop_container()
        provider.stop_container()
        assert adapter.deleted_pods == 1
        assert adapter.deleted_services == 1

    def test_close_deletes(self, provider, adapter):
        provider.start_container("echo-env:latest")
        provider.close()
        assert adapter.deleted_pods == 1
        assert provider._pod_created is False

    def test_context_manager_deletes_on_exit(self, adapter):
        with KubernetesProvider(namespace="openenv", _adapter=adapter) as started:
            started.start_container("echo-env:latest")
        assert adapter.deleted_pods == 1
        assert adapter.deleted_services == 1

    def test_is_container_provider(self, provider):
        assert isinstance(provider, ContainerProvider)

    def test_not_reexported_from_runtime_package(self):
        from openenv.core.containers import runtime

        assert "KubernetesProvider" not in runtime.__all__
        assert not hasattr(runtime, "KubernetesProvider")


class TestCleanup:
    def test_service_failure_deletes_only_the_pod(self, provider, adapter):
        adapter.fail_service = RuntimeError("service failed")
        with pytest.raises(RuntimeError, match="service failed"):
            provider.start_container("echo-env:latest")
        assert adapter.deleted_pods == 1
        assert adapter.deleted_services == 0
        assert provider._pod_created is False
        assert provider._name is None

    def test_pod_create_failure_deletes_nothing(self, provider, adapter):
        adapter.fail_pod = RuntimeError("Kubernetes create Pod name is already in use")
        with pytest.raises(RuntimeError, match="already in use"):
            provider.start_container("echo-env:latest")
        assert adapter.deleted_pods == 0
        assert adapter.deleted_services == 0

    def test_cleanup_failure_does_not_mask_original_error(self, provider, adapter):
        adapter.fail_service = RuntimeError("service failed")
        adapter.fail_delete_pod = RuntimeError("delete failed secret-body")
        with pytest.raises(RuntimeError, match="service failed") as exc_info:
            provider.start_container("echo-env:latest")
        assert "secret-body" not in str(exc_info.value)
        assert "delete failed" not in str(exc_info.value)
        assert adapter.deleted_pods == 1

    def test_failed_delete_can_be_retried(self, provider, adapter):
        provider.start_container("echo-env:latest")
        adapter.fail_delete_pod = RuntimeError("apiserver unavailable")
        with pytest.raises(RuntimeError, match="retry"):
            provider.stop_container()
        assert provider._pod_created is True
        adapter.fail_delete_pod = None
        provider.stop_container()
        assert provider._pod_created is False
        assert adapter.deleted_services == 1


class TestWaitForReady:
    def test_ready_when_running_and_health_200(self, provider):
        url = provider.start_container("echo-env:latest")
        with patch("requests.get", return_value=MagicMock(status_code=200)) as get:
            provider.wait_for_ready(url, timeout_s=5)
        assert get.call_args.args[0] == f"{url}/health"
        assert get.call_args.kwargs["proxies"] == {"http": None, "https": None}

    def test_failed_pod_raises_immediately_without_url(self, provider, adapter):
        url = provider.start_container("echo-env:latest")
        adapter.phase = "Failed"
        adapter.reason = "ImagePullBackOff"
        with (
            patch(
                "openenv.core.containers.runtime.kubernetes_provider.time.sleep"
            ) as sleep,
            pytest.raises(RuntimeError, match="ImagePullBackOff") as exc_info,
        ):
            provider.wait_for_ready(url, timeout_s=30)
        sleep.assert_not_called()
        assert url not in str(exc_info.value)
        assert "svc.cluster.local" not in str(exc_info.value)

    def test_unsafe_reason_is_omitted(self, provider, adapter):
        url = provider.start_container("echo-env:latest")
        adapter.phase = "Failed"
        adapter.reason = "secret-token\nImagePullBackOff"
        with pytest.raises(RuntimeError, match="phase Failed") as exc_info:
            provider.wait_for_ready(url, timeout_s=5)
        message = str(exc_info.value)
        assert "secret-token" not in message
        assert "ImagePullBackOff" not in message
        assert url not in message

    def test_timeout_does_not_leak_url(self, provider, adapter):
        import requests

        url = provider.start_container("echo-env:latest")
        adapter.phase = "Running"
        with (
            patch("requests.get", side_effect=requests.ConnectionError("refused")),
            patch(
                "openenv.core.containers.runtime.kubernetes_provider.time.time",
                side_effect=[0, 1, 100],
            ),
            pytest.raises(TimeoutError, match="did not become ready") as exc_info,
        ):
            provider.wait_for_ready(url, timeout_s=5)
        assert url not in str(exc_info.value)
        assert "svc.cluster.local" not in str(exc_info.value)

    def test_mismatched_base_url_is_rejected_without_echoing_it(self, provider):
        provider.start_container("echo-env:latest")
        evil = "http://evil.example/secret-token"
        with pytest.raises(ValueError, match="does not match") as exc_info:
            provider.wait_for_ready(evil, timeout_s=5)
        assert "secret-token" not in str(exc_info.value)
        assert "evil.example" not in str(exc_info.value)

    def test_trailing_slash_matches(self, provider):
        url = provider.start_container("echo-env:latest")
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            provider.wait_for_ready(url + "/", timeout_s=5)


def _install_fake_kubernetes(monkeypatch, *, incluster_fails: bool = False):
    """Install a minimal fake ``kubernetes`` module and return its call log."""
    calls: dict = {
        "incluster": 0,
        "kubeconfig": [],
        "create_pod": [],
        "create_service": [],
        "delete_pod": [],
        "delete_service": [],
        "read_namespace": [],
        "read_pod": [],
    }

    class ConfigException(Exception):
        pass

    class ApiException(Exception):
        def __init__(self, status=None, body=None):
            super().__init__("api failure")
            self.status = status
            self.body = body

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.ConfigException = ConfigException

    def load_incluster_config():
        calls["incluster"] += 1
        if incluster_fails:
            raise ConfigException("not in a cluster")

    def load_kube_config(config_file=None, context=None):
        calls["kubeconfig"].append((config_file, context))

    config_mod.load_incluster_config = load_incluster_config
    config_mod.load_kube_config = load_kube_config

    class _Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    client_mod = types.ModuleType("kubernetes.client")
    for name in (
        "V1ObjectMeta",
        "V1ContainerPort",
        "V1EnvVar",
        "V1ResourceRequirements",
        "V1Container",
        "V1LocalObjectReference",
        "V1PodSpec",
        "V1Pod",
        "V1ServicePort",
        "V1ServiceSpec",
        "V1Service",
    ):
        setattr(client_mod, name, type(name, (_Obj,), {}))

    class CoreV1Api:
        def read_namespace(self, name):
            calls["read_namespace"].append(name)
            if calls.get("namespace_status") == 404:
                raise ApiException(status=404, body="secret-namespace-body")
            return _Obj()

        def create_namespaced_pod(self, namespace, body):
            calls["create_pod"].append((namespace, body))
            if calls.get("create_pod_status") == 409:
                raise ApiException(status=409, body="secret-pod-body")
            return body

        def create_namespaced_service(self, namespace, body):
            calls["create_service"].append((namespace, body))
            return body

        def delete_namespaced_pod(self, name, namespace):
            calls["delete_pod"].append((name, namespace))
            if calls.get("delete_pod_status") == 404:
                raise ApiException(status=404, body="secret-delete-body")
            if calls.get("delete_pod_status") == 500:
                raise ApiException(status=500, body="secret-delete-body")

        def delete_namespaced_service(self, name, namespace):
            calls["delete_service"].append((name, namespace))

        def read_namespaced_pod(self, name, namespace):
            calls["read_pod"].append((name, namespace))
            waiting = _Obj(reason="ImagePullBackOff", message="secret-status-text")
            return _Obj(
                status=_Obj(
                    phase="Pending",
                    container_statuses=[
                        _Obj(state=_Obj(waiting=waiting, terminated=None))
                    ],
                )
            )

    client_mod.CoreV1Api = CoreV1Api
    exceptions_mod = types.ModuleType("kubernetes.client.exceptions")
    exceptions_mod.ApiException = ApiException
    client_mod.exceptions = exceptions_mod

    kubernetes_mod = types.ModuleType("kubernetes")
    kubernetes_mod.client = client_mod
    kubernetes_mod.config = config_mod

    monkeypatch.setitem(sys.modules, "kubernetes", kubernetes_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client.exceptions", exceptions_mod)
    return calls


class TestDefaultAdapter:
    def test_incluster_config_is_preferred(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        adapter.ensure_namespace()
        assert calls["incluster"] == 1
        assert calls["kubeconfig"] == []
        assert calls["read_namespace"] == ["openenv"]

    def test_falls_back_to_kubeconfig_outside_cluster(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch, incluster_fails=True)
        _DefaultKubernetesAdapter(namespace="openenv", kubeconfig=None, context=None)
        assert calls["incluster"] == 1
        assert calls["kubeconfig"] == [(None, None)]

    def test_explicit_kubeconfig_skips_incluster(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig="/tmp/kubeconfig", context="dev"
        )
        assert calls["incluster"] == 0
        assert calls["kubeconfig"] == [("/tmp/kubeconfig", "dev")]

    def test_create_translates_manifest_and_hides_conflict_body(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        calls["create_pod_status"] = 409
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        manifest = {
            "metadata": {
                "name": "openenv-echo",
                "namespace": "openenv",
                "labels": {"app.kubernetes.io/instance": "openenv-echo"},
            },
            "spec": {
                "restart_policy": "Never",
                "automount_service_account_token": False,
                "runtime_class_name": "gvisor",
                "image_pull_secrets": ["reg-cred"],
                "containers": [
                    {
                        "name": "env",
                        "image": "echo-env:latest",
                        "ports": [{"name": "http", "container_port": 8000}],
                        "env": [{"name": "TOKEN", "value": "supersecret"}],
                        "resources": {"limits": {"memory": "128Mi"}},
                    }
                ],
            },
        }
        with pytest.raises(RuntimeError, match="already in use") as exc_info:
            adapter.create_pod(manifest)
        assert "secret-pod-body" not in str(exc_info.value)
        assert "supersecret" not in str(exc_info.value)
        body = calls["create_pod"][0][1]
        assert body.spec.runtime_class_name == "gvisor"
        assert body.spec.automount_service_account_token is False
        assert body.spec.containers[0].image == "echo-env:latest"
        assert body.spec.containers[0].env[0].value == "supersecret"
        assert calls["create_pod"][0][0] == "openenv"

    def test_delete_404_is_idempotent_and_hides_body(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        calls["delete_pod_status"] = 404
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        adapter.delete_pod("openenv-echo")
        assert calls["delete_pod"] == [("openenv-echo", "openenv")]

    def test_delete_failure_hides_api_body(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        calls["delete_pod_status"] = 500
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        with pytest.raises(RuntimeError, match="Failed to delete Pod") as exc_info:
            adapter.delete_pod("openenv-echo")
        assert "secret-delete-body" not in str(exc_info.value)

    def test_missing_namespace_hides_api_body(self, monkeypatch):
        calls = _install_fake_kubernetes(monkeypatch)
        calls["namespace_status"] = 404
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        with pytest.raises(RuntimeError, match="does not exist") as exc_info:
            adapter.ensure_namespace()
        assert "secret-namespace-body" not in str(exc_info.value)

    def test_pod_status_keeps_reason_code_not_status_message(self, monkeypatch):
        _install_fake_kubernetes(monkeypatch)
        adapter = _DefaultKubernetesAdapter(
            namespace="openenv", kubeconfig=None, context=None
        )
        status = adapter.pod_status("openenv-echo")
        assert status["phase"] == "Pending"
        assert status["reason"] == "ImagePullBackOff"
        assert "secret-status-text" not in str(status)

    def test_missing_client_names_the_extra(self, monkeypatch):
        real_import = __import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "kubernetes" or name.startswith("kubernetes."):
                raise ImportError("blocked")
            return real_import(name, globals, locals, fromlist, level)

        for key in list(sys.modules):
            if key == "kubernetes" or key.startswith("kubernetes."):
                monkeypatch.delitem(sys.modules, key)
        monkeypatch.setattr("builtins.__import__", guarded)
        with pytest.raises(RuntimeError, match=r"openenv\[kubernetes\]"):
            _DefaultKubernetesAdapter(
                namespace="openenv", kubeconfig=None, context=None
            )


def test_module_import_does_not_import_kubernetes():
    source = kubernetes_provider.__file__
    assert source is not None
    # The optional client is imported inside the adapter constructor, not at
    # module import. Importing the provider module must not be what loads it
    # for callers who never construct the default adapter.
    import ast
    from pathlib import Path

    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    module_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = []
    for node in module_imports:
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any(
        name == "kubernetes" or name.startswith("kubernetes.") for name in imported
    )
