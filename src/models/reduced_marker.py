"""Small, persistent, explicitly approximate adapter to the reference kernel.

Retains all marker degrees of freedom; does not compress a million-particle
snapshot. Transport is 2D poloidal only, with constant D_perp, capped motion,
bilinear interpolation and lumped projection. ADAS PLT only (no PRB/CX) is
retained from B1; not a fully calibrated JOREK radiation closure.
"""
from copy import deepcopy
import numpy as np
import torch
from src.models.marker_state import MarkerCloud, advance_window, particle_window_schedule
from src.physics.spi_model import temperature_ev


def run_reduced_chain(chain, trace, *, dt_source_jorek, t_norm, central_density,
                      marker_state, u_start, u_end, central_mass=1.):
    from src.models.source_module import _adas_rates_np
    if marker_state is None:
        raise ValueError("marker transport requires persistent marker_state; use confirmed empty only before injection")
    if u_start.shape[0] != 1:
        raise ValueError("reference marker path supports batch=1; no implicit batched full-particle replay")
    if marker_state.feedback.layout != "node_dof0":
        raise ValueError("explicit reduce_checkpoint_marker adapter required for native full feedback")
    if marker_state.clock.startup_pending:
        raise ValueError("execute JOREK startup event before reduced marker propagation")
    if int(marker_state.cloud.alive.sum()) > 10000:
        raise ValueError("reference live-marker limit exceeded; calibrated reduction required")
    for name,value in {"t_norm":t_norm,"central_density":central_density,"central_mass":central_mass}.items():
        if name in marker_state.provenance and not np.isclose(marker_state.provenance[name],value,rtol=1e-12,atol=0):
            raise ValueError(f"marker state normalization mismatch: {name}")
    sampler, config = chain.sampler, chain.config
    dtype, device = u_start.dtype, u_start.device
    def tensor(a):
        return torch.as_tensor(a, dtype=dtype, device=device)
    def locate(x):
        loc = sampler.locate(tensor(x[:,:2]).unsqueeze(0))
        elm = np.where(loc.inside[0].cpu().numpy(), loc.element[0].cpu().numpy()+1, 0)
        return elm, torch.stack([loc.s[0],loc.t[0]],-1).cpu().numpy()
    state = deepcopy(marker_state)
    duration = float(torch.as_tensor(dt_source_jorek).reshape(-1)[0])*t_norm
    start, target = state.clock.sim_time_si, state.clock.sim_time_si+duration
    # Supplied endpoints are already available, not future truth. Caller owns
    # their times; reject a window not matching that recorded/model history.
    if abs(state.clock.field_start_si-start)>1e-14 or abs(state.clock.field_end_si-target)>1e-14:
        raise ValueError("marker window does not match available field interpolation history")
    previous = tensor(state.feedback.values).unsqueeze(0)
    def fields(time, cloud, ids):
        loc = sampler.locate(tensor(cloud.x[ids,:2]).unsqueeze(0))
        lam = (time-start)/duration
        if not -state.clock.step_rest_time/duration-1e-10 <= lam <= 1+1e-10:
            raise ValueError("field interpolation would extrapolate outside available endpoints")
        u = u_start*(1-lam)+u_end*lam
        f = sampler.sample(u, loc)[0]
        g = sampler.sample_gradient(u, loc)[0]
        feedback = sampler.sample(previous, loc)[0]
        fg = sampler.sample_gradient(previous, loc)[0]
        bhat = torch.stack([g[:,0,1], -g[:,0,0]],-1)
        bhat = bhat/bhat.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        fluid = torch.cat([f[:,6,None]*bhat/t_norm, torch.zeros((len(ids),1),dtype=dtype,device=device)],1)
        n_norm = central_density*1e20
        imp = (f[:,7]*n_norm*central_mass/20.).clamp_min(0.)
        if config.diff_diffusive_flux:
            log_grad = (fg[:,4,:]*n_norm-g[:,7,:]*n_norm*central_mass/20.)/(feedback[:,4]*n_norm-imp).clamp_min(1e16)[:,None]
        else:
            log_grad = g[:,7,:]*n_norm*central_mass/20./imp.clamp_min(1e12)[:,None]
        external = -config.diffusion_perp/t_norm*log_grad
        velocity = fluid.clone(); velocity[:,:2] += external
        # Registered reduced cap, not part of upstream JOREK pusher.
        limit = config.displacement_cap/state.clock.particle_dt_si
        velocity *= (limit/velocity.norm(dim=-1).clamp_min(limit)).unsqueeze(-1)
        raw_n = feedback[:,4].detach().cpu().numpy()
        n = raw_n.copy(); mask = n < 2e-4
        n[mask] = 1e-4+1e-4*np.exp((n[mask]-2e-4)/1e-4)
        zbar = np.clip(feedback[:,3].detach().cpu().numpy()/n,0,10)
        ne = ((f[:,4]-f[:,7]).detach().cpu().numpy()*n_norm).clip(1e16)+zbar*imp.detach().cpu().numpy()
        te = temperature_ev(f[:,8],central_density).clamp_min(1.)
        scd, acd, lz, ne_cm = _adas_rates_np(chain.tables,te,tensor(ne))
        radiation = lz.T*ne_cm[:,None]
        radiation[temperature_ev(f[:,8],central_density).detach().cpu().numpy()<=1.] = 0.
        return {"velocity": velocity.detach().cpu().numpy(), "fluid_velocity": fluid.detach().cpu().numpy(),
                "ne_si":ne,"scd":scd.T*1e-6,"acd":acd.T*1e-6,
                "radiation_w_per_atom":radiation}
    # Approximate deterministic quadrature births: fixed packets per shard per
    # trace record, all inserted before the window (upstream births are also
    # allocated before all substeps, with stochastic source rejection sampling).
    _, _, birth_gather_time = particle_window_schedule(state.clock,target)
    offsets = chain.birth_offsets.detach().cpu().numpy()
    positions=[]; weights=[]
    for rec in trace:
        shards=rec["shards"][0].detach().cpu().numpy()
        w=rec["ablation"][0].detach().cpu().numpy()*birth_gather_time/len(trace)/len(offsets)
        for j in np.flatnonzero(rec["alive"][0].cpu().numpy() & (w>0)):
            rz=shards[j,:2]+shards[j,chain.shard_index["radius"]]*offsets
            positions.extend(np.column_stack([rz,np.full(len(offsets),shards[j,chain.shard_index["phi"]])]))
            weights.extend([w[j]]*len(offsets))
    births=None
    if positions:
        births=MarkerCloud.empty(capacity=len(positions))
        births.x=np.asarray(positions); births.weight=np.asarray(weights)
        births.i_elm,births.st=locate(births.x)
        ids=np.flatnonzero(births.alive)
        births.V_prev[ids]=fields(start,births,ids)["fluid_velocity"]
    def sample(time,cloud,ids,feedback):
        return fields(time,cloud,ids)
    def deposit(cloud,ids,load):
        loc=sampler.locate(tensor(cloud.x[ids,:2]).unsqueeze(0))
        return sampler.deposit(tensor(load).unsqueeze(0),loc)[0].detach().cpu().numpy()
    measure=sampler.node_measure().detach().cpu().numpy()
    def project(rhs):
        return rhs/np.maximum(measure,max(1e-9,abs(measure.mean())*1e-6))[:,None]
    def reduced_push(x,old,v,dt):
        # Defined stationary extension for zero speed (upstream has 0/0).
        speed=np.linalg.norm(v,axis=1); prev=np.linalg.norm(old,axis=1)
        direction=np.divide(v,speed[:,None],out=np.zeros_like(v),where=speed[:,None]>0)
        before=np.divide(old,prev[:,None],out=direction.copy(),where=prev[:,None]>0)
        delta=(1.5*direction-.5*before)*speed[:,None]*dt
        result=x.copy(); result[:,:2]+=delta[:,:2]; result[:,2]+=delta[:,2]/x[:,0]
        return result
    updated,diagnostic=advance_window(state,target,sample=sample,locate=locate,
        deposit=deposit,project=project,potentials_ev=chain.tables.ionization_potentials_ev,
        t_norm=t_norm,central_density=central_density,births=births,grow=True,
        pusher=reduced_push)
    updated.provenance.update(t_norm=t_norm,central_density=central_density,central_mass=central_mass)
    updated.provenance["representation"]="all_slots_2d_bilinear_dof0_lumped_projection"
    updated.provenance["approximations"]=["fixed quadrature packets", "2D poloidal velocity",
        "constant D_perp", "motion cap", "zero-speed stationary extension", "PLT-only radiation",
        "bilinear fields", "lumped unfiltered projection", "detached reference physics"]
    cloud=updated.cloud; m=diagnostic["moments"]
    displacement=np.linalg.norm(cloud.x[:len(state.cloud.x),:2]-state.cloud.x[:,:2],axis=1)
    active=cloud.alive
    features=np.array([[displacement.mean() if len(displacement) else 0.,
                       (cloud.P_imp@np.arange(11))[active].mean() if active.any() else 0.,
                       (cloud.P_imp@np.arange(11)**2)[active].mean() if active.any() else 0.,
                       active.mean() if len(active) else 0., np.log1p(duration)]])
    features=np.clip(np.sign(features)*np.log1p(abs(features))/10,-1,1)
    return {"source":tensor(updated.feedback.values).unsqueeze(0),"state":updated,
            "features":tensor(features),"n_markers":len(cloud.weight),
            "gather_time_seconds":diagnostic["gather_time_seconds"],
            "cloud":{"positions":tensor(cloud.x[:,:2]).unsqueeze(0),
                     "alive":torch.as_tensor(active,device=device).unsqueeze(0),
                     "weights":tensor(cloud.weight).unsqueeze(0),
                     "fractions":tensor(cloud.P_imp).unsqueeze(0),
                     "V_prev":tensor(cloud.V_prev).unsqueeze(0),
                     "energy":tensor(diagnostic["energy"]).unsqueeze(0),
                     "rad":tensor(diagnostic["rad"]).unsqueeze(0),
                     "moment2":tensor(m[:,0]).unsqueeze(0),"moment1":tensor(m[:,1]).unsqueeze(0),
                     "time":tensor(m[:,2]).unsqueeze(0),
                     "displacement":tensor(displacement).unsqueeze(0)}}
