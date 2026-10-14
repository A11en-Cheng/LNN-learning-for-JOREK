"""Debug entry validates the seed before touching real data or creating output."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('seed',['-1',str(2**32)])
def test_smoke_rejects_invalid_seed_before_output_creation(tmp_path,seed):
    target=tmp_path/'no-output'
    result=subprocess.run([sys.executable,'-m','tools.smoke_native_autonomous',
        '--run','/nonexistent/data','--anchor','4110','--output-dir',str(target),
        '--seed',seed],capture_output=True,text=True)
    assert result.returncode==2
    assert '--seed must be' in result.stderr
    assert not target.exists()


def test_debug_smoke_help_exposes_seed_and_resolution():
    result=subprocess.run([sys.executable,'-m','tools.smoke_native_autonomous','--help'],
                          capture_output=True,text=True,check=True)
    assert '--seed' in result.stdout and '--birth-cells-per-width' in result.stdout
