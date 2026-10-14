"""Inspected non-stellarator, non-zonal H=1 particle projection operator.

Local second derivatives are analytic, including curved-map Hessian terms.
This is not the fluid nine-field boundary operator or an electron capacity.
"""
from copy import deepcopy
import numpy as np
from .native_state import configuration_digest


def validate_n0(spec,n_nodes):
    required={'filter_perp','filter_hyper','filter_parallel','F0','psi','field_time_si',
              'field_provenance','boundary','axis_nodes','fix_axis_nodes','apply_dirichlet'}
    if not isinstance(spec,dict) or set(spec)!=required:
        raise ValueError('complete explicit n0 projection configuration required')
    a=deepcopy(spec)
    for key in ('filter_perp','filter_hyper','filter_parallel','F0','field_time_si'):
        if not np.isscalar(a[key]) or not np.isfinite(a[key]):raise ValueError('finite n0 scalar required')
    if min(a[k] for k in ('filter_perp','filter_hyper','filter_parallel'))<0:
        raise ValueError('negative projection filter unsupported')
    if not isinstance(a['field_provenance'],str) or not a['field_provenance']:
        raise ValueError('projection initialization field provenance required')
    for flag in ('fix_axis_nodes','apply_dirichlet'):
        if type(a[flag]) is not bool:raise ValueError('explicit projection boolean flags required')
    psi=np.asarray(a['psi'])
    if psi.shape!=(4,1,n_nodes) or not np.isfinite(psi).all():
        raise ValueError('complete H1 initialization psi snapshot required')
    b=np.asarray(a['boundary']);axis=np.asarray(a['axis_nodes'])
    if b.shape!=(n_nodes,) or b.dtype.kind not in 'iu' or not np.isin(b,[0,1,2,3,4,5,9,11,15,19,20,21]).all():
        raise ValueError('unsupported projection boundary codes')
    if axis.shape!=(n_nodes,) or axis.dtype.kind!='b':raise ValueError('explicit axis-node mask required')
    a['psi']=psi.astype(np.float64);a['boundary']=b.astype(np.int64);a['axis_nodes']=axis.copy()
    return a


def _basis_two(s,t):
    def one(x):
        h=np.stack([1-3*x*x+2*x**3,3*x-6*x*x+3*x**3,
                    3*x*x-2*x**3,3*x*x-3*x**3],1).reshape(-1,2,2)
        d=np.stack([-6*x+6*x*x,3-12*x+9*x*x,
                    6*x-6*x*x,6*x-9*x*x],1).reshape(-1,2,2)
        dd=np.stack([-6+12*x,-12+18*x,6-12*x,6-18*x],1).reshape(-1,2,2)
        return h,d,dd
    ss=one(s);tt=one(t)
    result=[]
    for u,v in ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2)):
        a=np.empty((len(s),4,4))
        for i,(left,bottom) in enumerate(((0,0),(1,0),(1,1),(0,1))):
            for j,(dx,dy) in enumerate(((0,0),(1,0),(0,1),(1,1))):
                a[:,i,j]=ss[u][:,left,dx]*tt[v][:,bottom,dy]
        result.append(a)
    return result


def local_operator(mesh,e,g,spec):
    b,bs,bt,bss,bst,btt=[(a*mesh.size[:,:,e].T).reshape(16,16)
                         for a in _basis_two(g.s,g.t)]
    nodes=mesh.vertex[:,e]
    xy=mesh.coordinates[:,:,0,nodes].transpose(0,2,1).reshape(2,16).T
    # J[physical,local]; Hessian chain rule removes curved-map contribution.
    jac=np.stack((bs@xy,bt@xy),axis=-1)
    inv=np.linalg.inv(jac)
    local_grad=np.stack((bs,bt),axis=-1)
    grad=np.einsum('qki,qij->qkj',local_grad,inv)
    map_hess=np.stack((np.stack((bss@xy,bst@xy),-1),
                       np.stack((bst@xy,btt@xy),-1)),-2).transpose(0,1,3,2)
    # map_hess[q,physical,local,local]
    ref_hess=np.stack((np.stack((bss,bst),-1),np.stack((bst,btt),-1)),-2)
    corrected=ref_hess-np.einsum('qka,qaij->qkij',grad,map_hess)
    hess=np.einsum('qia,qkij,qjb->qkab',inv,corrected,inv)
    r=g.coordinates[:,0];vol=g.weights*g.jacobian*r
    lap=hess[:,:,0,0]+hess[:,:,1,1]+grad[:,:,0]/r[:,None]
    p=spec['psi'][:,0,nodes].T.reshape(16)
    pg=np.einsum('qka,k->qa',grad,p)
    bgrad=(grad[:,:,0]*pg[:,1,None]-grad[:,:,1]*pg[:,0,None])/r[:,None]
    bb2=(spec['F0']**2+(pg*pg).sum(1))/r
    if spec['filter_parallel']>0 and (bb2<=0).any():raise ValueError('singular parallel-filter magnetic metric')
    a=b.T@(vol[:,None]*b)
    a+=spec['filter_perp']*np.einsum('qka,qla,q->kl',grad,grad,vol)
    a+=spec['filter_hyper']*(lap.T@(vol[:,None]*lap))
    if spec['filter_parallel']>0:
        a+=spec['filter_parallel']*(bgrad.T@((vol/bb2)[:,None]*bgrad))
    a*=2*np.pi
    # JOREK replaces every local entry with matching global index BEFORE assembly.
    if spec['fix_axis_nodes']:
        for v,node in enumerate(nodes):
            if spec['axis_nodes'][node]:
                for degree in (2,3):
                    row=4*v+degree
                    matching=mesh.local_indices[e]==mesh.local_indices[e,row]
                    a[row,matching]=1e12
    return a


def penalties(mesh,spec):
    result=np.zeros(mesh.n_dofs)
    if spec['apply_dirichlet']:
        for node,code in enumerate(spec['boundary']):
            if code in (2,3,5,9):degrees=(0,2)
            elif code in (1,3,4,9):degrees=(0,1)
            else:degrees=()
            for d in degrees:result[mesh.inverse[d,node]]+=1e12
    return result


def fingerprint(spec):
    return dict(version='jorek-inspected-n0-v1',
                reconstruction_inputs={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items()},
                field_time_si=float(spec['field_time_si']),
                field_provenance=spec['field_provenance'],
                filters={k:float(spec[k]) for k in ('filter_perp','filter_hyper','filter_parallel','F0')},
                fix_axis_nodes=spec['fix_axis_nodes'],apply_dirichlet=spec['apply_dirichlet'],
                snapshot_and_boundary_digest=configuration_digest(spec),
                magnetic_policy='frozen_at_projection_creation',
                matrix_normalization='JOREK_full_2pi_matrix_unmultiplied_particle_rhs')
