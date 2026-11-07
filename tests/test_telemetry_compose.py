"""Check the effective opt-in configuration without starting Docker services."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "settings,sampler,ratio",
    [
        ({}, "parentbased_always_on", "1.0"),
        ({"OTEL_TRACES_SAMPLER": "always_off"}, "always_off", "1.0"),
        (
            {
                "OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
                "OTEL_TRACES_SAMPLER_ARG": "0.1",
            },
            "parentbased_traceidratio",
            "0.1",
        ),
    ],
)
def test_effective_sampler_and_loopback_ports(settings, sampler, ratio):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose CLI required; no daemon needed")
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env.update(settings)
    env.update(
        {
            "OTEL_BSP_MAX_QUEUE_SIZE": "64",
            "OTEL_BSP_MAX_EXPORT_BATCH_SIZE": "16",
            "OTEL_BSP_SCHEDULE_DELAY": "20",
            "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "0.5",
        }
    )
    env["OBS_JAEGER_UI_PORT"] = "16686"
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "compose.rag.yml",
            "-f",
            "compose.telemetry.yml",
            "--profile",
            "checks",
            "config",
            "--format",
            "json",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    services = json.loads(result.stdout)["services"]
    for name in ("api", "ui", "rag", "smoke"):
        assert services[name]["environment"]["OTEL_TRACES_SAMPLER"] == sampler
        assert services[name]["environment"]["OTEL_TRACES_SAMPLER_ARG"] == ratio
        for key in (
            "OTEL_BSP_MAX_QUEUE_SIZE",
            "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
            "OTEL_BSP_SCHEDULE_DELAY",
            "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
        ):
            assert services[name]["environment"][key] == env[key]
    jaeger = services["jaeger"]
    assert not jaeger.get("volumes")
    assert {port["target"] for port in jaeger["ports"]} == {16686}
    assert all(
        port["host_ip"] == "127.0.0.1"
        for service in services.values()
        for port in service.get("ports", [])
    )
    assert jaeger["image"] == "cr.jaegertracing.io/jaegertracing/jaeger:2.22.0"
    assert not jaeger.get("command")  # Use the pinned image's embedded memory configuration.
