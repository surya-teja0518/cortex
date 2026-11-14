from cortex.core.config import deep_merge, load_config


def test_deep_merge_dicts_merge_lists_replace():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    assert deep_merge(base, {"a": {"x": 9, "y": [3]}, "c": 3}) == {"a": {"x": 9, "y": [3]}, "b": 2, "c": 3}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 2}                         # input untouched


def test_local_file_overrides_and_can_be_disabled(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("linkedin_check: {enabled: false, max_per_run: 20}\nllm: {a: 1}\n")
    (tmp_path / "config.local.yaml").write_text("linkedin_check: {enabled: true}\n")
    monkeypatch.delenv("CORTEX_NO_LOCAL_CONFIG", raising=False)
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg["linkedin_check"] == {"enabled": True, "max_per_run": 20} and cfg["llm"] == {"a": 1}
    monkeypatch.setenv("CORTEX_NO_LOCAL_CONFIG", "1")
    assert load_config(str(tmp_path / "config.yaml"))["linkedin_check"]["enabled"] is False


def test_no_local_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.delenv("CORTEX_NO_LOCAL_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text("x: 1\n")
    assert load_config(str(tmp_path / "config.yaml")) == {"x": 1}
