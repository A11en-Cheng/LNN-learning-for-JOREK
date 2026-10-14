"""Frozen mesh class exercised through the public complete source seam."""
from copy import deepcopy
from pathlib import Path
import numpy as np


def test_alias_reference_mesh_matches_complete_source_advance_without_class_patch():
    from tools.benchmark_native_locator_geometry import load_reference_mesh
    from tests.test_native_coupling import native_fixture
    from tests.test_native_locator_latency import locator_mesh
    import src.models.native_coupling as coupling
    from tools.diagnose_native_real_chain import _array_digest
    mesh_type,evidence=load_reference_mesh(Path(coupling.__file__))
    assert coupling.NativeSourceMesh is not mesh_type
    geometry=locator_mesh()
    make=lambda cls:cls(geometry.coordinates,geometry.size,geometry.vertex,geometry.indices,
                       units=original.mesh.units,projection='dense')
    _,original,state,fields=native_fixture()
    fields=np.concatenate((fields,fields),axis=-1)
    state.feedback.values=np.zeros((5,4,1,8));state.feedback.deltas=np.zeros_like(state.feedback.values)
    results=[];rng=np.random.get_state()
    for cls in (mesh_type,coupling.NativeSourceMesh):
        mesh=make(cls)
        source=coupling.NativeMarkerSource(mesh,atomic=original.atomic,atomic_version=original.configuration['atomic'],
            potentials_ev=original.potentials_ev,F0=original.F0,diffusion_ju=original.diffusion_ju,spatial_backend='batched_v1')
        initial=deepcopy(state)
        initial.cloud.x[0,:2]=mesh.point(0,initial.cloud.st[0],fields)[2]
        np.random.set_state(rng)
        out,diag=source.advance(initial,1e-9,fields,fields)
        results.append((_array_digest(out),_array_digest(np.random.get_state()),
                        {k:_array_digest(v) for k,v in diag.items() if k!='execution'}))
    assert results[0]==results[1]
    assert evidence['sha256'] and evidence['alias']=='src.models._locator_reference'
