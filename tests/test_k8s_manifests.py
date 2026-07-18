"""Static checks for security-sensitive Kubernetes manifest wiring."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_config_is_generated_and_mounted_as_a_secret():
    kustomization = yaml.safe_load((ROOT / "k8s" / "kustomization.yaml").read_text())
    assert "configMapGenerator" not in kustomization

    generated_secrets = {entry["name"]: entry for entry in kustomization["secretGenerator"]}
    assert generated_secrets["missbot-config"]["files"] == ["config.yaml=config.yaml"]

    resources = list(yaml.safe_load_all((ROOT / "k8s" / "base.yaml").read_text()))
    deployment = next(resource for resource in resources if resource["kind"] == "Deployment")
    volumes = {volume["name"]: volume for volume in deployment["spec"]["template"]["spec"]["volumes"]}
    assert volumes["config-volume"]["secret"]["secretName"] == "missbot-config"
    assert "configMap" not in volumes["config-volume"]


def test_mem0_maintenance_cronjob_is_wired_to_the_cleanup_cli():
    kustomization = yaml.safe_load((ROOT / "k8s" / "kustomization.yaml").read_text())
    assert "maintenance.yaml" in kustomization["resources"]
    assert "maintenance-settings.yaml" in kustomization["resources"]

    settings = yaml.safe_load((ROOT / "k8s" / "maintenance-settings.yaml").read_text())
    assert settings["metadata"]["name"] == "missbot-maintenance-settings"
    assert settings["data"] == {
        "schedule": "17 4 * * *",
        "timeZone": "America/Los_Angeles",
    }
    replacement_fields = {
        (replacement["source"]["fieldPath"], replacement["targets"][0]["fieldPaths"][0])
        for replacement in kustomization["replacements"]
    }
    assert replacement_fields == {
        ("data.schedule", "spec.schedule"),
        ("data.timeZone", "spec.timeZone"),
    }

    cronjob = yaml.safe_load((ROOT / "k8s" / "maintenance.yaml").read_text())
    assert cronjob["kind"] == "CronJob"
    assert cronjob["metadata"]["name"] == "missbot-maintenance"
    assert cronjob["metadata"]["namespace"] == "misskey"
    assert cronjob["spec"]["timeZone"] == "America/Los_Angeles"
    assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"

    pod_spec = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["command"] == ["/app/.venv/bin/python", "-m", "bot.maintenance"]
    assert container["args"] == ["cleanup", "-c", "/config.yaml"]
    assert container["resources"]["limits"]["memory"] == "1Gi"
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    assert volumes["config-volume"]["secret"]["secretName"] == "missbot-config"
