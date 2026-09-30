# SPDX-License-Identifier: BSD-3-Clause
"""Live OpenShell isolation checks; opt-in errors must fail rather than skip."""

import json
import os
import socket
import textwrap

import pytest
from openenv.core.openenvd.policy import OpenEnvDConfig
from openenv.core.openenvd.runtime import Runtime


@pytest.fixture
def live_settings():
    gateway = os.environ.get("OPENSHELL_TEST_GATEWAY")
    if not gateway:
        pytest.skip("set OPENSHELL_TEST_GATEWAY to run live OpenShell integration")
    image = os.environ.get("OPENSHELL_TEST_IMAGE")
    if not image:
        pytest.fail(
            "OPENSHELL_TEST_IMAGE is required when a live gateway is configured"
        )
    return {"gateway": gateway, "image": image}


async def sandbox_python(runtime, source):
    backend = runtime.backend
    output = await backend._run(
        backend._command(
            "sandbox",
            "exec",
            "--name",
            backend.name,
            "--no-tty",
            "--no-login-shell",
            "--timeout",
            "30",
            "--",
            runtime.config.openshell.python,
            "-I",
            "-c",
            textwrap.dedent(source),
        ),
        timeout=45,
    )
    return json.loads(output)


@pytest.mark.integration
async def test_live_openshell_episode_and_principal_boundaries(
    live_settings, tmp_path, monkeypatch
):
    for principal in ("ORCHESTRATOR", "GRADER", "OBSERVER"):
        monkeypatch.setenv(
            f"OPENENVD_{principal}_TOKEN", f"test-only-{principal}-secret"
        )
    seed, assets = tmp_path / "seed", tmp_path / "assets"
    seed.mkdir()
    assets.mkdir(mode=0o700)
    (seed / "answer.txt").write_text("baseline")
    (assets / "solution.txt").write_text("private grading input")
    oracle = assets / "oracle.sh"
    oracle.write_text(
        "#!/bin/sh\n"
        'test "$(cat answer.txt)" = agent-change || exit 1\n'
        "printf 'local-only' > oracle-only.txt\n"
        "printf 'graded'\n"
    )
    oracle.chmod(0o700)
    config = OpenEnvDConfig.model_validate(
        {
            "enabled": True,
            "openshell": live_settings,
            "surfaces": {
                "orchestrator": {"allow_lifecycle": True},
                "agent": {"tools": ["echo_message"]},
                "grader": {
                    "tools": [
                        "grader.read_file",
                        "grader.fs_diff",
                        "grader.run_oracle",
                    ],
                    "fs_read": ["/workspace/**", "/openenvd/assets/**"],
                    "allow_privileged_exec": True,
                },
            },
            "privileged_assets": {"solution": "solution.txt", "oracle": "oracle.sh"},
        }
    )
    runtime = Runtime(
        config,
        "echo_env.server.echo_environment:EchoEnvironment",
        "openenv.core.env_server.mcp_types:CallToolAction",
        seed,
        asset_root=assets,
        timeout_s=300,
    )
    try:
        await runtime.start()
        reset = await runtime.reset({"episode_id": "first"})
        assert reset["done"] is False
        first_backend = runtime.backend
        first_id = first_backend.id
        result = await runtime.request(
            "step",
            {
                "type": "call_tool",
                "tool_name": "echo_message",
                "arguments": {"message": "live OpenShell echo"},
            },
        )
        assert "live OpenShell echo" in json.dumps(result)
        assert (await runtime.request("state"))["episode_id"] == "first"

        inaccessible = json.dumps(
            [str(assets), str(runtime.directory / "assets"), "/openenvd/assets"]
        )
        with socket.socket() as daemon_listener:
            daemon_listener.bind(("127.0.0.1", 0))
            daemon_listener.listen(1)
            daemon_port = daemon_listener.getsockname()[1]
            probe = await sandbox_python(
                runtime,
                f"""
                import errno, json, os, socket, urllib.error, urllib.request
                from pathlib import Path

                assert os.getuid() != 0 and os.getgid() != 0
                try:
                    os.setuid(0)
                except PermissionError:
                    pass
                else:
                    raise AssertionError('sandbox regained root')
                for path in {inaccessible}:
                    try:
                        Path(path).stat()
                    except OSError:
                        pass
                    else:
                        raise AssertionError('privileged asset path is visible')
                for principal in ('ORCHESTRATOR', 'GRADER', 'OBSERVER'):
                    assert f'OPENENVD_{{principal}}_TOKEN' not in os.environ
                try:
                    connection = socket.create_connection(('127.0.0.1', {daemon_port}), timeout=3)
                except OSError:
                    pass
                else:
                    connection.close()
                    raise AssertionError('daemon loopback connection succeeded')
                # The v0.1.2 connect broker returns EACCES for policy denial.
                # Timeouts and unreachable endpoints must not count as isolation.
                try:
                    with socket.create_connection(('1.1.1.1', 443), timeout=3):
                        pass
                except PermissionError as error:
                    assert error.errno == errno.EACCES
                else:
                    raise AssertionError('default-deny policy allowed public egress')
                base = 'http://127.0.0.1:8000'
                def rpc(method, params=None):
                    body = json.dumps({{'jsonrpc': '2.0', 'id': 1,
                                       'method': method, 'params': params or {{}}}}).encode()
                    request = urllib.request.Request(base + '/mcp', body,
                                                     {{'Content-Type': 'application/json'}})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        return json.load(response)
                tools = rpc('tools/list')['result']['tools']
                assert [tool['name'] for tool in tools] == ['echo_message']
                assert rpc('tools/call', {{'name': 'reset'}})['error']['code'] == -32602
                for route in ('/reset', '/state', '/mcp/grader', '/observe', '/ws'):
                    request = urllib.request.Request(base + route, b'{{}}',
                                                     {{'Content-Type': 'application/json'}})
                    try:
                        urllib.request.urlopen(request, timeout=5)
                    except urllib.error.HTTPError as error:
                        assert error.code == 404, (route, error.code)
                    else:
                        raise AssertionError('privileged route exists in sandbox')
                Path('/sandbox/workspace/answer.txt').write_text('agent-change')
                Path('/sandbox/workspace/created.txt').write_text('episode-only')
                try:
                    Path('/opt/openenv/agent-write.txt').write_text('forbidden')
                except PermissionError:
                    pass
                else:
                    raise AssertionError('installed runtime is writable')
                print(json.dumps({{'uid': os.getuid(), 'egress': 'denied'}}))
                """,
            )
        assert probe["uid"] != 0 and probe["egress"] == "denied"
        assert await runtime.read_file("/workspace/answer.txt") == "agent-change"
        assert (
            await runtime.read_file("/openenvd/assets/solution")
            == "private grading input"
        )
        assert await runtime.fs_diff() == [
            {"path": "/workspace/answer.txt", "kind": "modify"},
            {"path": "/workspace/created.txt", "kind": "create"},
        ]
        assert await runtime.run_oracle() == {
            "returncode": 0,
            "stdout": "graded",
            "stderr": "",
        }
        assert (
            await sandbox_python(
                runtime,
                "from pathlib import Path; import json; "
                "print(json.dumps(Path('/sandbox/workspace/oracle-only.txt').exists()))",
            )
            is False
        )
        assert (seed / "answer.txt").read_text() == "baseline"
        assert sorted(path.name for path in seed.iterdir()) == ["answer.txt"]

        await runtime.reset({"episode_id": "second"})
        assert runtime.backend.id != first_id
        assert await first_backend._owned(timeout=30) is None
        assert await runtime.read_file("/workspace/answer.txt") == "baseline"
        assert await runtime.fs_diff() == []
        assert (await runtime.request("state"))["episode_id"] == "second"
        assert (seed / "answer.txt").read_text() == "baseline"
    finally:
        backend = runtime.backend
        await runtime.close()
    assert await backend._owned(timeout=30) is None
