#!/usr/bin/env python3
"""Explicit-scope native causal window audit; no training or implicit scans."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_window_io import audit_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    result=audit_spec(json.loads(args.spec.read_text()),args.output_dir)
    print(json.dumps({k:result[k] for k in ('scope','window_counts','exclusions',
        'field_supervised_windows','conditional_source_windows','coupled_rollout_windows','parameter_status_counts','conclusion')},indent=2,ensure_ascii=False))


if __name__=='__main__': main()
