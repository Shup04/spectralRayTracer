#!/usr/bin/env python3
"""Offline synthetic STL render, CLI validation and byte reproducibility checks."""
import hashlib
from pathlib import Path
import re
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def synthetic_mesh(path):
    # After the existing -90 degree X rotation: a square in front of the camera.
    vertices = [(-20, 0, 10), (20, 0, 10), (20, 0, 50), (-20, 0, 50)]
    records = []
    for indices in [(0, 1, 2), (0, 2, 3)]:
        coords = [component for i in indices for component in vertices[i]]
        records.append(struct.pack('<12fH', 0, -1, 0, *coords, 0))
    path.write_bytes(b'Synthetic smoke square'.ljust(80, b'\0') + struct.pack('<I', 2) + b''.join(records))


def invoke(*args, success=True):
    result = subprocess.run([str(ROOT / 'spectral_tracer'), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return result


def read_ppm(path, width, height):
    values = path.read_text().split()
    assert values[:4] == ['P3', str(width), str(height), '255']
    channels = [int(x) for x in values[4:]]
    assert len(channels) == width * height * 3
    assert all(0 <= c <= 255 for c in channels)
    assert len(set(zip(channels[::3], channels[1::3], channels[2::3]))) > 1
    return channels


def main():
    with tempfile.TemporaryDirectory(prefix='spectral-smoke-') as directory:
        directory = Path(directory)
        mesh = directory / 'synthetic.stl'
        synthetic_mesh(mesh)
        for samples in (1, 2, 4, 5, 17, 65):
            output = directory / f'normal-{samples}.ppm'
            args = ['--mesh', mesh, '--width', 35, '--height', 19, '--samples', samples]
            result = invoke(*args, '--output', output)
            read_ppm(output, 35, 19)
            assert 'finite accumulation, clamped RGB' in result.stdout
            rays = int(re.search(r'primary rays: (\d+)', result.stdout)[1])
            assert 35 * 19 * min(4, samples) <= rays <= 35 * 19 * min(64, samples)
            repeat = directory / 'repeat.ppm'
            invoke(*args, '--output', repeat)
            assert repeat.read_bytes() == output.read_bytes(), 'fixed packet seeds must reproduce bytes'
        heatmap = directory / 'heatmap.ppm'
        invoke('--mesh', mesh, '--width', 35, '--height', 19, '--samples', 64,
               '--heatmap', '--output', heatmap)
        read_ppm(heatmap, 35, 19)
        # A sub-packet / sub-tile render must also be safe.
        tiny = directory / 'tiny.ppm'
        invoke('--mesh', mesh, '--width', 1, '--height', 1, '--samples', 1, '--output', tiny)
        assert len(tiny.read_text().split()) == 7
        for args in [('--width', '0'), ('--height', '-1'), ('--width', '1.5'),
                     ('--width', 'nan'), ('--height', '8193'), ('--samples', '65536'),
                     ('--samples', '0'), ('--samples', 'abc'), ('--mesh',),
                     ('--output', ''), ('--surprise',), ('--width', '8192', '--height', '8192')]:
            invoke(*args, success=False)
        invoke('--mesh', directory / 'missing.stl', success=False)
        for name, content in [('empty', b''), ('zero', b'0' * 80 + struct.pack('<I', 0)),
                              ('truncated', mesh.read_bytes()[:-1]),
                              ('nan', b'0' * 80 + struct.pack('<I', 1) + struct.pack('<12fH', *([float('nan')] * 12), 0)),
                              ('degenerate', b'0' * 80 + struct.pack('<I', 1) + bytes(50))]:
            broken = directory / (name + '.stl')
            broken.write_bytes(content)
            invoke('--mesh', broken, success=False)
        invoke('--mesh', mesh, '--width', 1, '--height', 1, '--samples', 1,
               '--output', directory, success=False)
        print('PASS: synthetic 35x19 renders at 1/2/4/5/17/65 spp budgets; repeat bytes; heatmap; 1x1; 19 invalid-input cases')
        print('65-budget RGB SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
