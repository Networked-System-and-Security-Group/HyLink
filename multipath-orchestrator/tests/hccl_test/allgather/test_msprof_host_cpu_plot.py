import csv
from pathlib import Path
import subprocess
import sys
from PIL import Image


def test_phase_cpu_plot_splits_measured_wall_not_marker_duration(tmp_path):
    with (tmp_path/'host_phase_cpu.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['case', 'iter', 'rank', 'phase', 'wall_us',
                                                 'cpu_us', 'voluntary', 'involuntary'])
        writer.writeheader()
        for rank in range(4):
            writer.writerow(dict(case='/logs/example', iter=9, rank=rank, phase='loop',
                                 wall_us=10000, cpu_us=5000, voluntary=0, involuntary=12))
    script = Path(__file__).with_name('plot_msprof_runtime.py')
    subprocess.run([sys.executable, str(script), str(tmp_path), '--iters', '9', '--host-cpu'], check=True)
    with Image.open(tmp_path/'host_phase_cpu_coverage.png') as image:
        assert image.getpixel((200, 130)) == (88, 154, 198)
        assert image.getpixel((900, 130)) == (221, 221, 221)
