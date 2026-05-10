from pathlib import Path
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_allgather_interference_launcher_uses_mpi_and_real_process_identity(tmp_path):
    capture = ROOT / "tests/unit/fixtures/mpi_launch_capture.sh"
    for name in ("mpirun", "hccl_allgather_with_pcie"):
        dest = tmp_path / name
        shutil.copy(capture, dest)
        dest.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"], NP="4",
               AMPCCL_PCCL_FORCE_ZERO_UNIQUE_ID="1", AMPCCL_SHM_PARAMS="1",
               AMPCCL_ALGO="static")
    result = subprocess.run(
        ["bash", str(ROOT / "tests/hccl_test/allgather/run.sh"),
         str(tmp_path / "hccl_allgather_with_pcie"), "--devices=4", "--iters=8"],
        env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "argv=-np 4 -genvall" in result.stdout
    assert "zero_id=0 shm=1 algo=static" in result.stdout
    assert "--devices=4 --iters=8" in result.stdout
