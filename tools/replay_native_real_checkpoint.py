"""Replay only the saved second real event; no repeated 40-substep burn-in."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tools.diagnose_native_real_chain import Stages,_initial_pairing,_array_digest,_rng_digest,_trace_public_stages
from tools.native_real_chain_builder import build_real


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=256,help='must match the original execution configuration')
    args=p.parse_args()
    if args.output.exists():raise ValueError('new replay evidence path required')
    original=json.loads((args.directory/'diagnostic.json').read_text())
    run=Path(original['run']);stages=Stages();torch.set_num_threads(2)
    random.seed(20261009);np.random.seed(20261009);torch.manual_seed(20261009)
    controls=_initial_pairing(run,original['anchor_index'],stages)
    options=SimpleNamespace(chunk_size=args.chunk_size,wall_seconds=float(original['wall_limit_seconds']),
        memory_mib=original['memory_limit_mib'],birth_max_proposals=original['birth_max_proposals'])
    model,source,shard,runtime,teacher=build_real(run,original['anchor_index'],controls,stages,options)
    from src.data.base_coupled import StepControl,Support,make_sequence
    from src.training.base_coupled import TorchRecomputedSourceEvents,load_session
    public=tuple(StepControl(c['start_si'],c['dt_si'],c['end_si'],c['origin'],c['source'],phase=c['phase']) for c in controls)
    scope=str(run);start=runtime.marker.clock.sim_time_si
    sequence=make_sequence(run=scope,task='self_generated',controls=public,nodes=(),burnin_steps=0,
        support=(Support(scope,start,public[-1].end_si,'registered_poststartup_and_psi_intervention',scope),),
        blocks=(Support(scope,start,public[-1].end_si,'diagnostic_only',scope),),
        state_evidence={'complete_state':'verified_reconstructed'},required_state=('complete_state',),model_reasons=(),
        parameters={'scope':scope,'qualification_granted':False,'recorded_psi_intervention':True})
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=teacher)
    payload=torch.load(args.directory/'first-event.pt',weights_only=False)
    print(json.dumps(dict(sequence_match=payload['sequence']==sequence.fingerprint,
        configuration_match=payload['adapter_configuration']==adapter.configuration_fingerprint)),flush=True)
    del payload
    loaded=load_session(args.directory/'first-event.pt',sequence,adapter)
    _trace_public_stages(model,source,shard,stages)
    out,_=adapter.advance(loaded.state,public[1]);state=out.runtime
    expected=np.load(args.directory/'event-2.npz')
    arrays=dict(values=state.fields_current,raw_deltas=state.fluid_deltas,memory=state.field_memory,
        feedback=state.marker.feedback.values,projection_deltas=state.marker.feedback.deltas,
        shard_positions=state.shards['positions'],shard_radius=state.shards['radius'],shard_ablation_atoms_s=state.shards['ablation_atoms_s'])
    diffs={k:dict(exact=np.array_equal(a,expected[k]),max_abs=float(np.max(np.abs(a-expected[k]))),
        changed=int(np.count_nonzero(a!=expected[k]))) for k,a in arrays.items()}
    original_row=next(r for r in original['stages'] if r['stage']=='committed_event_evidence' and r['status']=='completed' and r['ordinary_index']==2)
    hashes=dict(field_digest=_array_digest(state.fields_current),raw_deltas_digest=_array_digest(state.fluid_deltas),
        memory_digest=_array_digest(state.field_memory),cloud_digest=_array_digest(state.marker.cloud),
        feedback_digest=_array_digest(state.marker.feedback),shard_digest=_array_digest(state.shards))
    result=dict(array_differences=diffs,component_hashes={k:dict(exact=v==original_row[k],actual=v,expected=original_row[k]) for k,v in hashes.items()},
        complete_state_digest=_array_digest(out),rng_digest=_rng_digest(),optimizer_updates=0)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
