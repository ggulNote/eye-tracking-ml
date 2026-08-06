from pathlib import Path

from ggulnote_ml.settings import load_project_env, resolve_data_root


def test_dotenv_loads_local_data_root_without_overriding_shell(
    tmp_path: Path, monkeypatch
) -> None:
    dotenv_root = tmp_path / "dotenv-data"
    shell_root = tmp_path / "shell-data"
    (tmp_path / ".env").write_text(
        "GGULNOTE_DATA_ROOT=%s\nMLFLOW_TRACKING_URI=sqlite:///custom.db\n"
        % dotenv_root,
        encoding="utf-8",
    )
    monkeypatch.delenv("GGULNOTE_DATA_ROOT", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    loaded = load_project_env(tmp_path)
    assert loaded["GGULNOTE_DATA_ROOT"] == str(dotenv_root)
    assert resolve_data_root(tmp_path) == dotenv_root.resolve()

    monkeypatch.setenv("GGULNOTE_DATA_ROOT", str(shell_root))
    load_project_env(tmp_path)
    assert resolve_data_root(tmp_path) == shell_root.resolve()
