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
