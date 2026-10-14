#!/usr/bin/env python3
"""Actual phaseA labels/controls/masks, no runtime invention, no training."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from src.data.base_coupled import StepControl,Support,make_sequence,read_recorded_node
from src.data.namelist import parse_namelist
from src.data.output_contract import parse_macroscopic_times


def audit(spec):
    if spec['schema']!='native-base-record-audit-v1' or not spec['gamma_provenance']:
        raise ValueError('explicit base record audit/gamma provenance required')
    manifests=[];reads=[]
    for segment in spec['segments']:
        run=Path(segment['run_path']);table=parse_macroscopic_times(run/'macroscopic_vars.dat')
        labels={};meta={};tn=None
        # Read all actual support field records, including history, not just
        # local targets. No part/rawDP inferred from this field scan.
        indices=set(range(segment['complete_support_first'],segment['last']+1))
        indices.update((segment['block']['first'],segment['block']['last']))
        for index in sorted(indices):
            node,record=read_recorded_node(run/f'jorek{index:06d}.h5',gamma=spec['gamma'],
                projection_path=run/f'projections{index-1:06d}.h5' if index in segment['supervision_indices'] else None)
            tn=node.targets['field']['units'].t_norm_si
            if abs(node.time_si-table[index]*tn)>1e-14 or record['field_index']!=index:
                raise ValueError('internal time/index mismatch')
            labels[index]=node;meta[index]=record
        controls=[]
        for i in range(segment['first'],segment['last']):
            a=labels[i].time_si;b=labels[i+1].time_si
            dt=meta[i+1]['field_tstep_ju']*tn
            if abs(b-a-dt)>1e-14 or dt<=0:raise ValueError('ordinary segment requires actual positive base steps')
            controls.append(StepControl(a,dt,b,'recorded_schedule_replay',str(run/f'jorek{i+1:06d}.h5')))
        nl=run/'jorek_namelist_spitest';params=parse_namelist(nl.read_text())
        parameter_binding=dict(namelist_path=str(nl),sha256=hashlib.sha256(nl.read_bytes()).hexdigest(),
            ns_radius_min=params['ns_radius_min'],ns_radius_ratio=params['ns_radius_ratio'],
            full_parsed_namelist=params,effective_configuration_status=segment['state_evidence']['effective_configuration'])
        binding=segment['model_binding']
        # A record audit cannot certify an unexecuted model. An explicit bound
        # validator report can be imported without maintaining a stale feature list.
        model_reasons=['executing_adapter_not_bound'] if binding is None else list(binding['validation_reasons'])
        if binding is not None and (not binding.get('configuration_fingerprint') or not binding.get('evidence')):
            raise ValueError('model validation report requires executing configuration and evidence')
        block=segment['block']
        seq=make_sequence(run=run.name,task='self_generated',controls=controls,
            nodes=[labels[i] for i in segment['supervision_indices']],burnin_steps=segment['burnin_steps'],
            support=(Support(run.name,labels[segment['complete_support_first']].time_si,
                labels[segment['first']].time_si,'initialization_replay_history',segment['initialization_source']),),
            blocks=(Support(run.name,labels[block['first']].time_si,labels[block['last']].time_si,block['split'],'explicit_spec_block'),),
            state_evidence=segment['state_evidence'],required_state=segment['required_state'],model_reasons=model_reasons,
            parameters=parameter_binding)
        manifests.append(seq.manifest());reads.append(dict(run=run.name,record_reads=meta))
    return dict(schema='native-base-record-audit-result-v1',scope=spec['scope'],training=False,
        manifests=manifests,record_reads=reads,
        qualification_counts={k:sum(m['qualifications'][k]['eligible'] for m in manifests)
            for k in ('data_state','model_support','control_source','split_isolation','combined')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite existing evidence')
    result=audit(json.loads(args.spec.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['qualification_counts'],indent=2))


if __name__=='__main__':main()
