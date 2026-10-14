"""Real poststartup initial state for bounded diagnostics, never training.

Autonomous mode reads only initial full fields/shards. The diagnostic teacher
mode additionally reads subsequent explicit psi teachers.
Fixed prescribed weights and reference scales are not a trained predictor.
"""
from pathlib import Path
import numpy as np
import torch
import h5py


def build_real(run,anchor,controls,stages,args,*,feedback_mode='recorded_psi'):
    if feedback_mode not in ('recorded_psi','autonomous'):
        raise ValueError('explicit supported field feedback mode required')
    from src.data.native_psi_teacher import read_native_psi_teacher
    from src.data.marker_checkpoint import read_marker_snapshot
    from src.data.namelist import parse_namelist
    from src.data.native_state import read_native_update_mapping
    from src.models.native_coupling import NativeSourceMesh,NativeMarkerSource,NativeNeonAtomicRates,TorchMarkerLoads,CoupledNativeField
    from src.models.native_source_density import NativeShardDensityKernel
    from src.models.native_source_birth import NativeProductionBirths
    from src.models.native_birth_positions import NativeTabulatedBirthSampler
    from src.models.native_source_production import NativeProductionShardModel
    from src.models.native_shard_dynamics import InspectedNeonShardDynamics
    from src.models.native_psi_teacher import RecordedPsiFeedback,NativePsiTeacherShardModel
    from src.models.native_features import TorchNativeGauss,NativeFeaturePath,PARAMETER_SCHEMA
    from src.models.native_field_step import NativeFieldStep
    from src.models.native_field_boundary import NativeBoundaryLearningRows
    from src.models.native_temporal import CfCTimeCore
    from src.models.native_source_features import SourceFeatureScales
    from src.models.stencil_conv import build_fem_stencil_geometry
    from src.models.marker_feedback import MarkerFeedbackParameters
    from src.models.marker_state import MarkerRuntime,MarkerClock,ProjectionFeedback,ReferenceRollout,MarkerExecutionBudget,particle_window_schedule
    from src.models.native_state import NativeStateContext,configuration_digest
    from src.models.source_module import load_neon_tables
    from src.physics.adas_closure import parse_adf11,ADAS_DIR
    from src.data.native_parameters import CONSUMER_BY_BASE
    run=Path(run);initial_path=run/f'jorek{anchor+1:06d}.h5'
    params=parse_namelist((run/'jorek_namelist_spitest').read_text())
    def flag(name,default):
        value=params.get(name,default)
        if isinstance(value,bool):return value
        if str(value).lower() not in ('.t.','.true.','.f.','.false.'):
            raise ValueError('explicit boolean configuration required: '+name)
        return str(value).lower() in ('.t.','.true.')
    # Defaults independently checked in preset_parameters/read_num_profiles;
    # reject branches this reference implementation cannot execute.
    dp=np.asarray(params['d_perp'],dtype=float)
    if (len(dp)<6 or dp[1]!=0 or dp[5]!=0 or params.get('d_perp_file','none')!='none'
        or flag('diff_diffusive_flux',False) or int(params['spi_abl_model'])!=3
        or int(params['spi_rkt_model'])!=6 or not flag('nonlocal_abl',False)
        or params.get('ns_delta_minor_rad',0.)!=0):
        raise ValueError('unsupported target diffusion/source branch; no silent zero or local fallback')
    with stages.track('teacher_records_read' if feedback_mode=='recorded_psi' else 'initial_field_records_read'):
        import json
        registry=json.loads((Path(__file__).resolve().parents[1]/'data/restart_events.json').read_text())['runs']
        registered=tuple(registry[run.name]['restart_events'])
        if anchor+1 not in registered:raise ValueError('registered poststartup initialization required')
        read=lambda path:read_native_psi_teacher(path,lcfs_restart_index=anchor,registered_restart_indices=registered)
        initial=read(initial_path)
        imported=read(run/f'jorek{anchor:06d}.h5')
        records=([initial]+[read(Path(c['source'])) for c in controls]
                 if feedback_mode=='recorded_psi' else None)
    with stages.track('all_slot_cloud_read') as evidence:
        cloud,particle_time,metadata=read_marker_snapshot(run/f'part{anchor:05d}.h5')
        evidence.update(marker_slots=len(cloud.weight),live_markers=int(cloud.alive.sum()),metadata=metadata)
    with h5py.File(initial_path,'r') as f:
        scalar=lambda k:float(np.asarray(f[k][...]).ravel()[0])
        u=f['values'][...];raw=f['deltas'][...];dt_previous=scalar('tstep')*initial.units.t_norm_si
        F0=scalar('F0');eta=scalar('eta')
        locator_mode=getattr(args,'marker_locator','native')
        if locator_mode not in ('native','jorek_nearby_v1'):
            raise ValueError('unsupported explicit marker locator mode')
        neighbours=None
        if locator_mode=='jorek_nearby_v1' or getattr(args,'marker_transport','native_fixed')=='local_subcycling_boundary_v2':
            if 'neighbours' not in f:
                raise ValueError('saved neighbours required; no inferred topology fallback')
            neighbours=np.asarray(f['neighbours'][...])
        s=dict(positions=np.column_stack([f[k][...] for k in ('spi_R_arr','spi_Z_arr','spi_phi_arr')]),
            velocities=np.column_stack([f[k][...] for k in ('spi_Vel_R_arr','spi_Vel_Z_arr','spi_Vel_RxZ_arr')]),
            radius=f['spi_radius_arr'][...],ablation_atoms_s=f['spi_abl_arr'][...],species=f['spi_species_arr'][...],
            injector_phi=f['spi_phi_init_arr'][...],ids=np.arange(int(scalar('n_spi_tot'))),
            time_si=initial.time_si,positions_coordinate_system='RZphi',velocities_coordinate_system='injector_R_Z_RxZ',
            source_volume_m3=f['spi_vol_arr'][...],source_volume_time_si=initial.time_si,
            source_volume_provenance=str(initial_path)+'::spi_vol_arr')
    with h5py.File(run/f'jorek{anchor:06d}.h5','r') as f:
        previous=f['values'][...];imported_dt=float(f['tstep'][0])*initial.units.t_norm_si
    with h5py.File(run/f'projections{anchor:06d}.h5','r') as f:
        p=f['values'][...];projection_time=float(f['t_now'][0])
    if p.any() or abs(projection_time-particle_time)>1e-14:
        raise ValueError('registered original zero projection and actual particle time required')
    particle_dt=imported_dt/max(1,int(imported_dt/float(params['tstep_particles'])))
    feedback=ProjectionFeedback(p,np.zeros_like(p),projection_time,'native_full','jorek_restart_zero_projection',None)
    marker=MarkerRuntime(cloud,feedback,MarkerClock(particle_time,0.,particle_dt,particle_time,initial.time_si),
        dict(fluid_control_origin='recorded_schedule_replay',initialization='registered_poststartup_import',
             imported_field_record_time_si=imported.time_si,imported_field_sha256=imported.source_sha256,
             startup_field_sha256=initial.source_sha256,initial_projection='registered_actual_zero',
             birth_topology='single_rank_distribution_not_production_MPI_slot_assignment'))
    spec=dict(filter_perp=float(params['filter_perp_n0']),filter_hyper=float(params['filter_hyper_n0']),
        filter_parallel=float(params['filter_par_n0']),F0=F0,psi=imported.psi,
        field_time_si=imported.time_si,field_provenance=str(run/f'jorek{anchor:06d}.h5')+' projection creation',
        boundary=initial.boundary,axis_nodes=np.zeros(len(initial.boundary),bool),
        fix_axis_nodes=False,apply_dirichlet=True)
    with stages.track('filtered_and_density_mesh_build'):
        mesh=NativeSourceMesh(initial.coordinates,initial.size,initial.vertex,initial.indices,
            units=initial.units,projection='sparse',max_elements=5000,projection_n0=spec)
        density_mesh=NativeSourceMesh(initial.coordinates,initial.size,initial.vertex,initial.indices,
            units=initial.units,projection='sparse',max_elements=5000)
    tables=load_neon_tables()
    rates=NativeNeonAtomicRates(tables,radiation_mode='plt_prb',prb=parse_adf11(ADAS_DIR/'prb96_ne.dat'))
    budget=MarkerExecutionBudget(max_slots=len(cloud.weight),max_live=len(cloud.weight),chunk_size=args.chunk_size,
        max_particle_steps=10000,max_seconds=args.wall_seconds,max_rss_bytes=args.memory_mib*1024**2)
    transport_mode=getattr(args,'marker_transport','native_fixed')
    if transport_mode not in ('native_fixed','local_subcycling_v1','local_subcycling_boundary_v2'):
        raise ValueError('unsupported explicit marker transport mode')
    transport_policy=None
    if transport_mode in ('local_subcycling_v1','local_subcycling_boundary_v2'):
        from src.models.marker_transport import LocalSubcycling,initialize_transport_history
        transport_policy=LocalSubcycling(boundary_exit=transport_mode=='local_subcycling_boundary_v2')
        marker=initialize_transport_history(marker,transport_policy,previous_dt_si=particle_dt)
    source=NativeMarkerSource(mesh,atomic=rates,atomic_version=rates.version,potentials_ev=tables.ionization_potentials_ev,
        F0=F0,diffusion_ju=float(dp[0]),time_sampling='jorek_raw_deltas_v1',field_static=True,execution_budget=budget,
        spatial_backend=getattr(args,'spatial_backend','scalar'),transport_policy=transport_policy,
        saved_neighbours=(neighbours if transport_mode=='local_subcycling_boundary_v2' else None))
    if locator_mode=='jorek_nearby_v1':
        from src.models.native_nearby_locator import NativeNearbyMarkerSource
        source=NativeNearbyMarkerSource(source,neighbours=neighbours)
    density=NativeShardDensityKernel(density_mesh,ns_radius_min=float(params['ns_radius_min']),
        ns_radius_ratio=float(params['ns_radius_ratio']),ns_deltaphi=float(params['ns_deltaphi']),
        spi_num_vol=flag('spi_num_vol',True),ns_delta_minor_rad=0.,
        injection_time_si=float(params['t_ns'])*initial.units.t_norm_si,toroidal_angles=(0.,))
    birth_mode=getattr(args,'birth_position_mode','native_probability_n0')
    if birth_mode=='native_probability_n0':
        position_sampler=NativeTabulatedBirthSampler(geometry='axisymmetric_n0',n_tor=1,
            cells_per_width=getattr(args,'birth_cells_per_width',64),
            max_cells=getattr(args,'birth_max_cells',16_000_000))
    elif birth_mode=='native_probability_periodic':
        position_sampler=NativeTabulatedBirthSampler(geometry='periodic_phi',n_tor=1,
            cells_per_width=getattr(args,'birth_cells_per_width',8),phi_cells=getattr(args,'birth_phi_cells',64),
            max_cells=getattr(args,'birth_max_cells',2_000_000))
    elif birth_mode=='native_rejection_reference':position_sampler=None
    else:raise ValueError('explicit supported native birth position mode required')
    birth=NativeProductionBirths(density,physical_particles=float(params['spi_quantity']),
        n_particle_slots=len(cloud.weight),n_cpu=1,rank=0,R_geo=float(params['r_geo']),n_plane=1,
        max_proposals=args.birth_max_proposals,position_sampler=position_sampler)
    dynamics=InspectedNeonShardDynamics(mesh,material_density_m3=float(params['pellet_density'])*1e20,
        F0=F0,R_geo=float(params['r_geo']),magnetic_reduction=flag('spi_abl_mag_reduction',False),
        nonlocal_abl=True,ns_radius_min=float(params['ns_radius_min']),ns_radius_ratio=float(params['ns_radius_ratio']))
    base=NativeProductionShardModel(dynamics,density,birth,n_flux=int(params['n_flux']),n_table=int(params['n_nonlocal_array']))
    if feedback_mode=='recorded_psi':
        teacher=RecordedPsiFeedback(records,mesh=mesh);shard=NativePsiTeacherShardModel(base,teacher)
    else:
        from src.models.native_equilibrium import NativeEquilibriumProvider
        teacher=NativeEquilibriumProvider.from_initial_record(mesh,initial)
        shard=base
    shard.configuration['historical_nonlocal_compatibility']='unified_current_ES_single_aux_not_historical_rank_lag_double_aux'
    with stages.track('joint_nine_field_model_build'):
        evaluator=TorchNativeGauss(initial.coordinates,initial.size,initial.vertex,units=initial.units)
        statistics=dict(version='native-gauss-train-statistics-v1',geometry=evaluator.geometry_digest,
            training_support=[['explicit_unit_references_NOT_training_fit',anchor+1]],
            scenario_enabled=True,physics_enabled=True,value_center=np.zeros(9),value_scale=np.ones(9),
            gradient_scale=np.ones((9,2)),scenario_scale=np.array([1000.,1e9]),magnetic_scale=np.ones(2))
        features=NativeFeaturePath(evaluator,statistics)
        gamma=initial.units.gamma
        boundary_parameters=dict(gamma=gamma,gamma_sheath_i=(gamma-1)*(float(params['gamma_i_stangeby'])-1-gamma),
            gamma_sheath_e=(gamma-1)*(float(params['gamma_e_stangeby'])-1),
            density_reflection=float(params.get('density_reflection',0.)),imp_reflection=float(params.get('imp_reflection',0.)),
            min_sheath_angle=float(params.get('min_sheath_angle',1.)),F0=F0,T_min=float(params['t_min']),
            visco_par_heating=float(params['visco_par_heating']))
        boundary=NativeBoundaryLearningRows(vertex=initial.vertex,boundary=initial.boundary,indices=initial.indices,
            freeboundary=flag('freeboundary',False),RMP_on=flag('rmp_on',False),no_mach1_bc=flag('no_mach1_bc',False),
            use_simple_bnd_types=flag('use_simple_bnd_types',False),bc_natural_open=flag('bc_natural_open',False),
            mach_one_bnd_integral=flag('mach_one_bnd_integral',False),vpar_smoothing=flag('vpar_smoothing',False),
            grid_to_wall=flag('grid_to_wall',False),n_wall_blocks=int(params['n_wall_blocks']),
            parameters=boundary_parameters,parameter_provenance=initial.namelist_path+' + inspected preset_parameters defaults + model502 initialise_parameters sheath conversion')
        mapping=read_native_update_mapping(initial_path,keep_n0_const=False,boundary_policy='learned_unverified')
        mapping.boundary_rows=boundary
        field=NativeFieldStep(build_fem_stencil_geometry(initial.vertex,initial.coordinates[:,0,0].T),mapping,
            units=initial.units,state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),hidden=8,n_layers=1,n_iters=2,
            feature_path=features,temporal_core=CfCTimeCore(8,backbone_units=11,time_scale_ju=1.))
        response=TorchMarkerLoads(mesh,parameters=MarkerFeedbackParameters(initial.units.n_norm_m3/1e20,
            initial.units.central_mass,gamma,'Ne',10,'element_matrix',True,True,True),temperature_floor=float(params['t_min']))
        scales=SourceFeatureScales.anchored_physical_reference(units=initial.units,origin_si=initial.time_si,
            gather_scale_si=dt_previous,event_scale_si=dt_previous,origin_provenance='explicit_initial_event_not_training_fit')
        model=CoupledNativeField(field,response,source_configuration=source.configuration,
            shard_configuration=shard.configuration,source_scales=scales)
        with torch.no_grad():
            for ordinal,parameter in enumerate(model.parameters()):
                index=torch.arange(parameter.numel(),dtype=parameter.dtype).reshape(parameter.shape)
                parameter.copy_(.02*torch.sin(index*.17+ordinal*.31))
            model.field.decoder.weight.fill_(1e-8)
            model.field.decoder.bias.copy_(torch.linspace(2e-7,4e-7,36,dtype=torch.float64))
        model.eval()
    rows=[]
    for name,group,unit in PARAMETER_SCHEMA:
        value=eta if name=='eta' else float(params[name])
        rows.append(dict(name=name,group=group,unit=unit,value=value,status='declared',available_at='initialization',
            consumer=CONSUMER_BY_BASE[name],provenance=dict(kind='h5' if name=='eta' else 'namelist',
            source=str(initial_path) if name=='eta' else initial.namelist_path)))
    context=NativeStateContext(initial.units,(0,),dict(mesh={'x':initial.coordinates,'size':initial.size,
        'vertex':initial.vertex+1,'index':initial.indices},boundary={'policy':boundary.policy,'native':{'boundary':initial.boundary}},
        physics={'feature_parameter_rows':rows},numerics={'scheme':params['time_evol_scheme']},model=model.checkpoint_configuration))
    if neighbours is not None:
        context.configuration['mesh']['neighbours']=neighbours.copy()
    with stages.track('poststartup_diagnostics'):
        branch=shard.diagnostic_fields(u,initial.time_si)
        s=shard.refresh_diagnostics(s,u,feedback,teacher(branch,initial.time_si),provenance='registered poststartup actual U and installed zero aux')
    runtime=ReferenceRollout(marker,u,previous,s,initial.time_si,anchor+1,tuple(c['dt_si'] for c in controls),
        fluid_dt_previous_si=dt_previous,fluid_deltas=raw,time_evol_scheme=params['time_evol_scheme'],
        native_context=context,field_memory=np.zeros((mesh.n_nodes,8)))
    runtime.validate_fluid_history();model.reference_solver(context)
    with stages.track('poststartup_ready') as evidence:
        evidence.update(startup_pending=False,diffusion_ju=source.diffusion_ju,ns_radius_min=density.minimum,
            field_feedback_mode=feedback_mode,
            birth_position_mode=birth_mode,birth_configuration=birth.configuration,
            projection=mesh.configuration['projection'],field_shape=list(u.shape),source_field_static=source.field_static,
            actual_first_particle_steps=particle_window_schedule(marker.clock,initial.time_si)[1],
            model_weights_unchanged=True,model_parameter_count=sum(p.numel() for p in model.parameters()),
            model_weights_digest=configuration_digest({k:v.detach().numpy() for k,v in model.state_dict().items()}),
            approximations=['single_rank_birth_distribution_not_MPI_or_PCG32_replay',
                'unified_current_ES_single_aux_not_historical_rank_lag_double_aux',
                'open_boundary_response_learned_not_explicit_solver','untrained_prescribed_nonzero_network',
                'unit_reference_scales_not_fitted_training_statistics','numpy_source_and_shards_stop_gradient'])
    return model,source,shard,runtime,teacher
