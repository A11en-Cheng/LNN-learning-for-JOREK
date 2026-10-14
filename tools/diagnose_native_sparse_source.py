#!/usr/bin/env python3
"""Bounded real mesh projection reproduction; no particle replay or training."""
import argparse,json,time,resource
from pathlib import Path
import h5py
import numpy as np
from src.models.native_coupling import NativeSourceMesh
from src.models.native_state import NativeUnits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('refuse to overwrite evidence')
    selected=[('phaseA_dataset_1d22_ang15_massloss_neon_v400',4110),
              ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',2703)]
    rows=[]
    for name,index in selected:
        path=Path('data/EHL-2')/name/f'jorek{index:06d}.h5'
        start=time.perf_counter()
        with h5py.File(path) as f:
            scalar=lambda k:float(np.asarray(f[k][()]).ravel()[0])
            units=NativeUnits(scalar('t_norm'),scalar('central_density')*1e20,scalar('central_mass'),5/3)
            mesh=NativeSourceMesh(f['x'][...],f['size'][...],f['vertex'][...]-1,f['index'][...],
                units=units,projection='sparse',max_elements=5000)
        build=time.perf_counter()-start
        # Exact constant polynomial, not generated with the implementation's basis.
        coefficients=np.zeros((1,4,1,mesh.n_nodes));coefficients[:,0]=1.
        start=time.perf_counter()
        values=[];point_error=0.
        for e,g in enumerate(mesh.gauss):
            v,grad,*_=mesh.point(e,np.array([.37,.61]),coefficients)
            point_error=max(point_error,float(abs(v[0]-1)),float(abs(grad).max()))
            values.append(np.ones((16,1)))
        rhs=mesh.weak_load(np.asarray(values));out=mesh.project(rhs)
        error=float(abs(out-coefficients).max())
        assert error<1e-8,(name,error)
        assert point_error<1e-9,(name,point_error)
        rows.append(dict(run=name,index=index,elements=len(mesh.gauss),nodes=mesh.n_nodes,
            unique_dofs=mesh.n_dofs,nnz=mesh.mass.nnz,
            sparse_matrix_bytes=mesh.mass.data.nbytes+mesh.mass.indices.nbytes+mesh.mass.indptr.nbytes,
            factor_nnz=mesh._factor.L.nnz+mesh._factor.U.nnz,
            build_factor_seconds=build,point_evaluate_assemble_solve_seconds=time.perf_counter()-start,
            constant_coefficient_max_error=error,constant_value_gradient_max_error=point_error,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scope='unfiltered_H1_projection_not_production_source_or_BC_validation'))
        del mesh
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'training':False,'records':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
