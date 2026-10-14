"""Model502 H1 Dirichlet increment rows; open/sheath BC remain unverified.

Inspected mod_boundary_conditions:200–364. This applies ideal fixed update
rows, not finite-penalty matrix solves, Mach1 or natural-open operators.
"""
import numpy as np
import torch
from torch import nn
from .native_state import configuration_digest


class NativeDirichletRows(nn.Module):
    def __init__(self,*,vertex,boundary,indices,freeboundary,RMP_on,no_mach1_bc,
                 use_simple_bnd_types,bc_natural_open):
        super().__init__()
        flags=dict(freeboundary=freeboundary,RMP_on=RMP_on,no_mach1_bc=no_mach1_bc,
            use_simple_bnd_types=use_simple_bnd_types,bc_natural_open=bc_natural_open)
        if any(type(v) is not bool for v in flags.values()) or freeboundary or RMP_on or use_simple_bnd_types:
            raise ValueError('unsupported fluid Dirichlet configuration')
        b=np.asarray(boundary);v=np.asarray(vertex);ids=np.asarray(indices)
        if (b.ndim!=1 or b.dtype.kind not in 'iu' or not np.isin(b,[0,1,2,3,4,5,9,11,12,15,19,20,21]).all()
            or ids.shape!=(4,len(b)) or ids.dtype.kind not in 'iu' or (ids<=0).any()
            or v.ndim!=2 or v.shape[0]!=4 or v.dtype.kind not in 'iu' or (v<0).any() or (v>=len(b)).any()):
            raise ValueError('complete supported native boundary/vertex/indices required')
        mask=np.zeros((9,4,1,len(b)),bool)
        for element in v.T:
            for iv,node in enumerate(element):
                if b[node]==0:continue
                for other in ((iv+1)%4,(iv+3)%4):
                    if b[element[other]]==0:continue
                    product=(iv+1)*(other+1)
                    degree=1 if product in (2,12) else 2 if product in (6,4) else None
                    if degree is None:raise ValueError('unsupported non-edge boundary pair')
                    # model502/mod_model_settings:39 identifies var_rhon=8,
                    # the impurity-density field; it is included at BC:344.
                    fields=range(9) if b[node] in (2,3,12,20,21) or no_mach1_bc else (0,1,2,3)
                    for f in fields:mask[f,(0,degree),0,node]=True
        # Constraints belong to unique solver DOFs, including all shared aliases.
        _,inverse=np.unique(ids,return_inverse=True)
        inverse=inverse.ravel()
        for f in range(9):
            blocked=np.zeros(inverse.max()+1,bool)
            np.logical_or.at(blocked,inverse,mask[f].ravel())
            mask[f]=blocked[inverse].reshape(4,1,len(b))
        self.register_buffer('mask',torch.from_numpy(mask))
        self.configuration=dict(version='model502-h1-dirichlet-rows-v2_rhon8',flags=flags,
            boundary=b.tolist(),indices=ids.tolist(),vertex=v.tolist(),
            mask_digest=configuration_digest(mask),
            coverage='dirichlet_rows_only_open_boundary_unverified',
            approximation='ideal_zero_increment_instead_of_finite_1e12_penalty')

    def forward(self,increment):
        if increment.shape!=self.mask.shape:raise ValueError('complete native increment required for BC')
        return increment.masked_fill(self.mask,0.)


class NativeBoundaryLearningRows(NativeDirichletRows):
    """Known fixed rows plus explicit inputs for coupled open-boundary learning.

    Node Mach1 is a coupled linearized penalty (Ti, Te, Phi, vpar), not
    vpar=cs. Natural terms are weak loads and Jacobian entries. Only the
    inherited fixed increment rows are projected. This module supplies their
    exact applicability and physical parameters to the joint field network;
    it does not claim to execute the open-boundary matrix or enforce its solution.
    Sources: inspected model502 boundary_conditions:227-248,367-370;
    matrix/construct_matrix_mod:85-185. H1/cubic only, as the field module.
    """
    feature_width=56
    policy='dirichlet_rows_open_learned_v1'
    parameter_names=('gamma','gamma_sheath_i','gamma_sheath_e','density_reflection',
                     'imp_reflection','min_sheath_angle','F0','T_min','visco_par_heating')

    def __init__(self, *, mach_one_bnd_integral, vpar_smoothing, grid_to_wall, n_wall_blocks,
                 parameters, parameter_provenance, **kwargs):
        super().__init__(**kwargs)
        if any(type(flag) is not bool for flag in (mach_one_bnd_integral,vpar_smoothing,grid_to_wall)):
            raise ValueError('explicit Mach1 integral and wall flags required')
        if type(n_wall_blocks) is not int or n_wall_blocks<0:
            raise ValueError('explicit wall-block count required')
        if set(parameters)!=set(self.parameter_names) or not parameter_provenance:
            raise ValueError('complete boundary parameters and provenance required')
        scalars=np.array([parameters[k] for k in self.parameter_names],dtype=float)
        if (not np.isfinite(scalars).all() or parameters['gamma']<=1
            or not 0<=parameters['min_sheath_angle']<=90 or parameters['T_min']<=0
            or parameters['visco_par_heating']<0):
            raise ValueError('invalid physical boundary parameters')
        b=np.asarray(kwargs['boundary']);vertex=np.asarray(kwargs['vertex'])
        mach=np.zeros((4,len(b)),bool);natural=mach.copy()
        side={1:2,11:2,5:3,15:3,2:3,12:2,4:2}
        for element in vertex.T:
            for iv,node in enumerate(element):
                if not b[node]:continue
                for other in ((iv+1)%4,(iv+3)%4):
                    adjacent=element[other]
                    if not b[adjacent]:continue
                    degree=1 if (iv+1)*(other+1) in (2,12) else 2
                    if (not kwargs['no_mach1_bc'] and not mach_one_bnd_integral
                        and b[node] in (1,3,4,5,9,11,15,19)
                        and not (b[node]==3 and b[adjacent]==2)):
                        mach[[0,degree],node]=True
                # Natural-integral selection uses the directed cyclic edge,
                # with wall-side priority/skip rules, not the nodal BC mask.
                adjacent=element[(iv+1)%4];b1,b2=int(b[node]),int(b[adjacent])
                if not b2 or not kwargs['bc_natural_open']:continue
                if grid_to_wall and n_wall_blocks>0:
                    s1,s2=side.get(b1,0),side.get(b2,0)
                    if not (s1 or s2) or {s1,s2}=={2,3}:continue
                    degree=1 if 2 in (s1,s2) else 2
                else:
                    if any(b1 in group and b2 in group for group in ({1,3},{1,9},{4,9},{1,4})):
                        degree=1
                    elif b1 in (5,9) and b2 in (5,9) or b1 in (2,3) and b2 in (2,3):
                        degree=2
                    else:continue
                natural[np.ix_([0,degree],[node,adjacent])]=True
        self.register_buffer('mach_rows',torch.from_numpy(mach))
        self.register_buffer('natural_rows',torch.from_numpy(natural))
        self.register_buffer('physical_parameters',torch.from_numpy(scalars))
        self.register_buffer('learning_flags',torch.tensor([kwargs['no_mach1_bc'],mach_one_bnd_integral,
            vpar_smoothing],dtype=torch.float64))
        self.configuration.update(version='model502-h1-boundary-learning-v1',
            policy=self.policy,coverage='fixed_rows_and_open_applicability_inputs',
            weak_terms='joint_network_learned_not_explicit_matrix_or_hard_projection',
            mach_one_bnd_integral=mach_one_bnd_integral,vpar_smoothing=vpar_smoothing,
            grid_to_wall=grid_to_wall,n_wall_blocks=n_wall_blocks,
            parameters=dict(parameters),parameter_provenance=parameter_provenance,
            features='36_fixed_mask_4_nodal_mach_mask_4_natural_trace_mask_9_native_parameters_3_flags',
            mach_rows=mach.tolist(),natural_rows=natural.tolist())

    def learning_features(self, values):
        n=values.shape[-1]
        fixed=self.mask.permute(3,0,1,2).reshape(n,36)
        params=self.physical_parameters.expand(n,-1)
        return torch.cat((fixed,self.mach_rows.T,self.natural_rows.T,params,
            self.learning_flags.expand(n,-1)),dim=1).to(values)
