import pytest

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot


@pytest.mark.parametrize("save_interval", [32, 0])
def test_mopd_launch_configuration(monkeypatch, tmp_path, save_interval):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_moss_tts_local.py")
    call_entrypoint(
        module,
        "execute",
        dict(
            objective="mopd",
            teachers="speech=http://teacher:19001/score_actions",
            student_score_endpoint="http://student-score:19002/score_actions",
            save_interval=save_interval,
        ),
        sandbox=tmp_path,
    )
    output = format_recording(recording, sandbox=tmp_path)
    name = "execute_mopd" if save_interval else "execute_mopd_no_save"
    assert_matches_snapshot(
        REPO_ROOT / f"tests/snapshots/launch_scripts/py/scripts/run_moss_tts_local.py/{name}.txt", output, name
    )
    assert "--moss-local-student-score-endpoint" in output
    assert "--custom-rm-path" not in output


@pytest.mark.parametrize("components", ["wer=1", "wer=1.00", "wer=1,sim=0"])
def test_equivalent_wer_settings_keep_the_original_launcher_snapshot(monkeypatch, tmp_path, components):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_moss_tts_local.py")
    call_entrypoint(module, "execute", {"reward_components": components}, sandbox=tmp_path)
    output = format_recording(recording, sandbox=tmp_path)
    assert_matches_snapshot(
        REPO_ROOT / "tests/snapshots/launch_scripts/py/scripts/run_moss_tts_local.py/execute.txt", output, "execute"
    )
