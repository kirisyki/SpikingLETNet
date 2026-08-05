from __future__ import annotations

import torch

from squat_comparison.resume_compat import install_cpu_checkpoint_restore


def test_resume_compat_installs_cpu_mapping(monkeypatch) -> None:
    import squat_comparison.checkpointing as checkpointing
    import squat_comparison.training as training

    captured = {}

    def fake_load(path, **kwargs):
        captured["map_location"] = kwargs["map_location"]
        return {"loader_generator_state": torch.Generator().get_state()}

    monkeypatch.setattr(checkpointing, "load_checkpoint", fake_load)
    monkeypatch.delattr(training.load_checkpoint, "_squat_cpu_restore", raising=False)
    install_cpu_checkpoint_restore()
    payload = training.load_checkpoint("checkpoint", model=object(), map_location="cuda")
    assert captured["map_location"] == "cpu"
    generator = torch.Generator()
    generator.set_state(payload["loader_generator_state"])

