"""Summarize bounded motion replays, with dimensionless sensitivity probes.

Forensics only. Finite differences use the production point evaluator and the
already-validated velocity decomposition; they are not an AB stability proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.benchmark_native_lookup import _json
from tools.diagnose_native_nearby_motion import load_capture, select_cloud, sample_detail


def relative(a,b,origin=None):
    a=np.asarray(a);b=np.asarray(b)
    denominator=np.linalg.norm(b if origin is None else b-np.asarray(origin))
    return float(np.linalg.norm(b-a)/denominator) if denominator else None


def analyze(directory):
    inputs={}
    def read(name):
        path=directory/name;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    full=read('capture-inner.json');selected=read('replay-selected.json')
    capture=directory/'capture-inner.pickle'
    inputs[str(capture)]=hashlib.sha256(capture.read_bytes()).hexdigest()
    payload=load_capture(capture);source=payload['source'];rows=[]
    for slot in (600505,900629,400633):
        index=selected['original_slots'].index(slot)
        a=selected['traces']['baseline'];b=selected['traces']['nearby']
        k=next(i for i,(r,s) in enumerate(zip(a,b)) if not np.array_equal(r['velocity'][index],s['velocity'][index]))
        r,s=a[k],b[k]
        row=dict(slot=slot,first_velocity_difference_substep=k,
            first_velocity_relative_l2=relative(s['velocity'][index],r['velocity'][index]),
            min_baseline_native_rho_imp=min(t['detail']['field'][index][7] for t in a),
            min_nearby_native_rho_imp=min(t['detail']['field'][index][7] for t in b),
            normalized_density_floor=1e12/(source.mesh.units.n_norm_m3*source.mesh.units.central_mass/20))
        c=read(f'dt32-slot{slot}.json');d=read(f'dt64-slot{slot}.json')
        row['cross_locator_dt32']=c['motion'];row['cross_locator_dt64']=d['motion']
        row['dt32_to_dt64']={}
        for mode in ('baseline','nearby'):
            trace32,trace64=c['traces'][mode],d['traces'][mode]
            row['dt32_to_dt64'][mode]=dict(
                displacement_relative_l2=relative(trace32[-1]['next_x'],trace64[-1]['next_x'],trace64[0]['x']),
                velocity_increment_relative_l2=relative(trace32[-1]['velocity'],trace64[-1]['velocity'],trace64[0]['V_prev']),
                velocity_state_relative_l2=relative(trace32[-1]['velocity'],trace64[-1]['velocity']))
        rows.append(row)
    primary=read('red-slot600505.json');sensitivities=[]
    cloud=select_cloud(payload['initial'],np.array([0,0,0,0])).cloud
    for k in (6,35,44,48,50,75,77):
        point=primary['traces']['baseline'][k];st=np.array(point['st'][0]);element=point['element'][0]
        cloud.i_elm[:]=element
        _,jac=source.mesh._location_geometry(element-1,st)
        estimates=[]
        for eps in (1e-6,1e-7,1e-8):
            cloud.st[:]=np.array([st+[eps,0],st-[eps,0],st+[0,eps],st-[0,eps]])
            detail=sample_detail(source,cloud,np.arange(4),payload['current'],payload['initial'].feedback)
            velocity=detail['exb']+detail['parallel'];velocity[:,:2]+=detail['diffusion']
            derivative_st=np.column_stack(((velocity[0]-velocity[1])/(2*eps),(velocity[2]-velocity[3])/(2*eps)))
            derivative_rz=derivative_st[:2]@np.linalg.inv(jac)
            estimates.append(dict(local_epsilon=eps,dt_velocity_jacobian_norm=primary['particle_dt_si']*np.linalg.norm(derivative_rz,2)))
        gradient=np.array(point['detail']['gradient'][0][7])*source.mesh.units.n_norm_m3*source.mesh.units.central_mass/20
        scale=max(point['detail']['ni_raw'][0],1e12)/np.linalg.norm(gradient)
        sensitivities.append(dict(substep=k,jacobian_condition=float(np.linalg.cond(jac)),
            normalized_density=point['detail']['field'][0][7],finite_difference_estimates=estimates,
            diffusion_displacement_over_density_scale=primary['particle_dt_si']*np.linalg.norm(point['detail']['diffusion'][0])/scale))
    nodal_evidence={str(element):dict(vertices=source.mesh.vertex[:,element-1].tolist(),
        rho_imp_dof0=payload['current'][7,0,0,source.mesh.vertex[:,element-1]].tolist())
        for element in (3354,3370)}
    refined=read('refined-guarded-slot600505.json')
    return dict(schema='native-nearby-motion-analysis-v1',training=False,production_qualified=False,
        input_sha256=inputs,full_cloud_motion=full['motion'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_replay_exact={mode:all(v['array_equal'] for v in values.values())
            for mode,values in selected['full_cloud_replay_comparison'].items()},
        particles=rows,representative_sensitivity=sensitivities,
        representative_element_node_values=nodal_evidence,guarded_baseline_refinement_motion=refined['motion'],
        sensitivity_scope='local_velocity_RZ_derivative_finite_difference_not_full_history_map_or_formal_stability_bound')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    with args.output.open('x') as stream:
        result=analyze(args.directory)
        json.dump(result,stream,default=_json,indent=2,allow_nan=False)
    print(json.dumps(result['particles'],default=_json))


if __name__=='__main__':main()
