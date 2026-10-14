"""Observe the real complete source seam on refinement-limit failure only."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import validate_native_nearby_history as validation
from tools.diagnose_native_nearby_motion import sample_detail
from tools.benchmark_native_lookup import _json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from src.models.native_coupling import NativeMarkerSource
    from src.models.marker_state import advance_window
    original=NativeMarkerSource.advance
    findings=[]
    def observed(self,state,target,previous,current,**options):
        policy=self.transport_policy
        def event(*a,**kw):
            sample=kw['sample']
            def record(t,cloud,ids,feedback):
                result=sample(t,cloud,ids,feedback)
                score=state.clock.particle_dt_si*result['transport_rate_si']
                requested=np.maximum(policy.min_level,np.ceil(np.log2(np.maximum(score/policy.max_step_indicator,1.))))
                positions=np.flatnonzero((score>policy.trigger_indicator)&(requested>policy.max_level))
                if len(positions):
                    chosen=ids[positions[:16]]
                    sampler=options.get('time_sampler')
                    fields=(sampler.fields_at(t) if sampler is not None else
                        previous+(current-previous)*((t-state.clock.field_start_si)/(state.clock.field_end_si-state.clock.field_start_si)))
                    detail=sample_detail(self,cloud,chosen,fields,feedback)
                    findings.append(dict(time_si=t,target_si=target,slots=chosen.tolist(),
                        lifetime=cloud.i_life[chosen].tolist(),born_this_window=(cloud.i_life[chosen]!=state.cloud.i_life[chosen]).tolist(),
                        score=score[positions[:16]],requested_level=requested[positions[:16]],
                        initial_sticky_level=a[0].transport_history.level[chosen],
                        x=cloud.x[chosen],st=cloud.st[chosen],i_elm=cloud.i_elm[chosen],
                        previous_velocity=cloud.V_prev[chosen],current_velocity=result['velocity'][positions[:16]],
                        ni_raw=detail['ni_raw'],ni_denominator=detail['ni_denominator'],
                        grad_ni=detail['gradient'][:,7]*self.mesh.units.n_norm_m3*self.mesh.units.central_mass/20,
                        old_state_live=state.cloud.alive[chosen],old_state_x=state.cloud.x[chosen]))
                return result
            kw['sample']=record
            return advance_window(*a,**kw)
        with patch('src.models.native_coupling.advance_window',event):
            return original(self,state,target,previous,current,**options)
    # Existing bounded writer hashes its own dependencies and records failure;
    # observation returns sample arrays unmodified, with no solver retry.
    with patch.object(NativeMarkerSource,'advance',observed):
        result=validation.main(['--run-index','1','--marker-transport','local_subcycling_v1',
            '--output',str(args.output)])
    path=args.output.with_name(args.output.stem+'-failure-detail.json')
    with path.open('x') as stream:json.dump(dict(observer='readonly_sample_no_physics_change',
        findings=findings,training=False),stream,default=_json,indent=2,allow_nan=False)
    print(json.dumps(dict(detail_file=str(path),findings=findings),default=_json),flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
