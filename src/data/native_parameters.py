"""Grouped parameter inventory, not a flat model condition vector.

Reuses the recorded H5 -> production namelist -> default chain. Unconfirmed
fallbacks and unresolved units remain visible but cannot enter current input.
Consumers are declarations for later integration; none is implemented here.
"""
import numpy as np

# Unit declarations inherited from native_run's existing audited subset.
# Remaining units are deliberately unresolved: do not invent SI conversions.
from src.data.native_run import _UNIT_BY_PARAMETER, _GROUP_BY_PARAMETER

# Local verified overlay; do not alter the legacy flat parameter interface.
UNIT_OVERLAY={'pellet_density':'10^20 m^-3','central_density':'10^20 m^-3','ZK_par_max':None}

NUMERIC={'tstep','t_now','time_evol_theta','time_evol_zeta'}
SCENARIO={'n_spi','pellet_density','T_0','Ti_0','Te_0','particlesource','ellip'}
CONSUMER_BY_BASE={
    'eta':'magnetic_diffusion_coefficient_pending',
    'visco':'poloidal_viscosity_coefficient_pending',
    'visco_par':'parallel_viscosity_coefficient_pending',
    'visco_par_par':'parallel_viscosity_coefficient_pending',
    'ZK_i_par':'ion_parallel_heat_coefficient_pending',
    'ZK_e_par':'electron_parallel_heat_coefficient_pending',
    'D_perp':'density_radial_profile_coefficient_pending',
    'ZK_i_perp':'ion_radial_heat_profile_pending',
    'ZK_e_perp':'electron_radial_heat_profile_pending',
    'T_0':'background_temperature_condition_pending',
    'Ti_0':'ion_background_temperature_condition_pending',
    'Te_0':'electron_background_temperature_condition_pending',
    'central_density':'density_normalization_context',
    'central_mass':'mass_normalization_context',
    'F0':'magnetic_geometry_closure_pending',
    'R_domm':'geometry_context_pending',
    'sqrt_mu0_rho0':'unit_time_conversion_context',
    't_norm':'unit_time_conversion_context',
    'n_spi':'pellet_count_condition_encoder_pending',
    'pellet_density':'pellet_material_condition_encoder_pending',
    'gamma_i_stangeby':'ion_sheath_boundary_coefficient_pending',
    'gamma_e_stangeby':'electron_sheath_boundary_coefficient_pending',
    'particlesource':'background_density_source_pending',
    'ellip':'equilibrium_shape_condition_pending',
    'ZK_par_max':'parallel_heat_cap_pending',
    'ZKpar_T_dependent':'temperature_dependent_heat_closure_pending',
    'Ti_min_ZKpar':'ion_heat_floor_pending',
    'Te_min_ZKpar':'electron_heat_floor_pending',
    'T_min':'temperature_floor_pending',
    'T_min_neg':'negative_temperature_branch_pending',
    'rho_min':'density_floor_pending',
    'rho_min_neg':'negative_density_branch_pending',
    'ZK_e_par_neg':'electron_negative_heat_branch_pending',
    'ZK_i_par_neg':'ion_negative_heat_branch_pending',
    'D_par':'main_density_parallel_diffusion_pending',
    'D_par_imp':'impurity_density_parallel_diffusion_pending',
    'freeboundary':'boundary_configuration_pending',
    'resistive_wall':'wall_boundary_configuration_pending',
    'bc_natural_open':'open_boundary_configuration_pending',
    'n_open':'open_boundary_configuration_pending',
    'n_limiter':'limiter_boundary_configuration_pending',
}
FLAGS={'bc_natural_open','freeboundary','resistive_wall','n_open','n_limiter','ZKpar_T_dependent'}


def route_parameters(payload, *, source, namelist_source, confirmed_defaults=()):
    if not source: raise ValueError('parameter source provenance required')
    names=payload['names']; vals=payload['vector']; origins=payload['provenance']
    if len(names)!=len(vals): raise ValueError('parameter length mismatch')
    rows=[]
    for name,value in zip(names,vals):
        base=name.split('[')[0]
        if base not in _GROUP_BY_PARAMETER: raise ValueError(f'parameter {name} is not registered')
        origin=origins.get(name)
        if origin not in ('h5','namelist','default','derived'):
            raise ValueError(f'unknown parameter provenance: {name}')
        if origin=='namelist' and not namelist_source:
            raise ValueError('namelist provenance required')
        if not np.isfinite(value): raise ValueError('nonfinite parameter')
        group='numeric_time' if base in NUMERIC else 'scenario' if base in SCENARIO else 'physical_coefficient'
        unit=UNIT_OVERLAY.get(base,_UNIT_BY_PARAMETER.get(base,'1' if base in FLAGS or base in ('time_evol_theta','time_evol_zeta') else None))
        status='declared'
        if unit is None: status='unresolved_unit'
        if origin=='default' and name not in confirmed_defaults: status='unconfirmed_default'
        consumer=CONSUMER_BY_BASE.get(base,'time_contract')
        if group=='numeric_time':
            consumer='time_contract_recompute_from_dt_pair' if origin=='derived' else 'time_contract'
            status='audit_only'  # do not smuggle recorded future time into inference
        rows.append(dict(name=name,group=group,unit=unit,value=float(value),
            provenance=dict(kind=origin,source=source if origin=='h5' else namelist_source if origin=='namelist' else 'legacy_JOREK_default_chain'),
            available_at='initialization' if status=='declared' else 'unavailable_until_confirmed',
            normalizer='train_only_group_affine_pending' if group=='scenario' else 'explicit_unit_conversion_pending',
            consumer=consumer,supported_range='observed_only_no_extrapolation_claim',status=status))
    # Duplicate names in the legacy surface are retained for audit, never routed.
    counts={n:names.count(n) for n in names}
    for r in rows:
        if counts[r['name']]>1:
            r['status']='duplicate_legacy_entry'; r['available_at']='unavailable_until_confirmed'
    return rows


def parameter_variation(split_rows):
    """Within-split variability and exact affine collinearity, audit only."""
    result={}
    for split,samples in split_rows.items():
        names=sorted({r['name'] for rows in samples for r in rows})
        columns={}; gaps=[]
        for n in names:
            vals=[[r['value'] for r in rows if r['name']==n] for rows in samples]
            if not vals or any(len(x)!=1 for x in vals): gaps.append(n); continue
            columns[n]=np.asarray([x[0] for x in vals])
        constant=[n for n,c in columns.items() if np.ptp(c)==0]
        varying=[n for n in columns if n not in constant]
        pairs=[]
        for i,a in enumerate(varying):
            for b in varying[i+1:]:
                corr=float(np.corrcoef(columns[a],columns[b])[0,1])
                if abs(corr)>1-1e-12: pairs.append([a,b,corr])
        result[split]=dict(samples=len(samples),constant_unidentifiable=constant,
            varying={n:dict(min=float(columns[n].min()),max=float(columns[n].max())) for n in varying},
            affine_collinear=pairs,missing_or_duplicate=gaps,
            note='two settings or collinear variation cannot identify independent parameter effects')
    return result
