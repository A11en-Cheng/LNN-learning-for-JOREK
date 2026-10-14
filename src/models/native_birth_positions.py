"""Event-local, rejection-free birth positions (explicit numerical experiment).

The table represents the *clipped total* source, not clipped individual shards.
Cells have trilinear acceptance in R**2/Z/phi and exact cylindrical volume. Sampling
is exact for that table, not for the continuous Gaussian source. No FEM location,
field sampling, marker weights, or cross-event cache lives in this module.
"""
from dataclasses import dataclass
from copy import deepcopy
from math import ceil, erf, pi, sqrt

import numpy as np
from scipy.special import erf as array_erf
from .native_birth_acceptance import NATIVE_BIRTH_ACCEPTANCE,native_birth_threshold


@dataclass(frozen=True)
class PreparedBirthPositions:
    r_edges: np.ndarray
    z_edges: np.ndarray
    phi_edges: np.ndarray
    probabilities: np.ndarray
    cdf: np.ndarray
    acceptance_nodes: np.ndarray
    diagnostics: dict

    def sample(self, count, *, rng=None):
        """Four uniforms per position; no physical-source rejection or lookup."""
        if type(count) is not int or count < 0:
            raise ValueError('nonnegative integer birth count required')
        rng = np.random if rng is None else rng
        draws = np.asarray(rng.random((count, 4)))
        if draws.shape != (count, 4) or not np.isfinite(draws).all() or (draws < 0).any() or (draws >= 1).any():
            raise ValueError('four finite uniforms in [0,1) per birth required')
        cells = np.searchsorted(self.cdf, draws[:, 0], side='right')
        ir, iz, ip = np.unravel_index(cells, self.probabilities.shape)
        c000=self.acceptance_nodes[ir,iz,ip];c001=self.acceptance_nodes[ir,iz,ip+1]
        c010=self.acceptance_nodes[ir,iz+1,ip];c011=self.acceptance_nodes[ir,iz+1,ip+1]
        c100=self.acceptance_nodes[ir+1,iz,ip];c101=self.acceptance_nodes[ir+1,iz,ip+1]
        c110=self.acceptance_nodes[ir+1,iz+1,ip];c111=self.acceptance_nodes[ir+1,iz+1,ip+1]
        # Rosenblatt transform of a positive trilinear cell: each conditional
        # density is linear, so every inverse is closed form, not iterative.
        ru=_linear_inverse((c000+c001+c010+c011)/4,
                           (c100+c101+c110+c111)/4,draws[:,1])
        zu=_linear_inverse(((1-ru)*(c000+c001)+ru*(c100+c101))/2,
                           ((1-ru)*(c010+c011)+ru*(c110+c111))/2,draws[:,2])
        pu=_linear_inverse((1-zu)*((1-ru)*c000+ru*c100)+zu*((1-ru)*c010+ru*c110),
                           (1-zu)*((1-ru)*c001+ru*c101)+zu*((1-ru)*c011+ru*c111),draws[:,3])
        r0, r1 = self.r_edges[ir], self.r_edges[ir + 1]
        z0, z1 = self.z_edges[iz], self.z_edges[iz + 1]
        p0, p1 = self.phi_edges[ip], self.phi_edges[ip + 1]
        return np.column_stack((np.sqrt(r0*r0 + ru*(r1*r1-r0*r0)),
                                z0 + zu*(z1-z0),p0 + pu*(p1-p0)))


def _linear_inverse(a0,a1,uniform):
    """Invert integral of a0+(a1-a0)*x using cancellation-safe quadratic root."""
    y=uniform*(a0+a1)/2
    denominator=a0+np.sqrt(np.maximum(a0*a0+2*(a1-a0)*y,0.))
    # Zero-density conditional at an exactly zero uniform is measure zero;
    # retain a bounded deterministic coordinate without division by zero.
    result=np.divide(2*y,denominator,out=np.zeros_like(y),where=denominator>0)
    return np.clip(result,0.,1.)


class NativeTabulatedBirthSampler:
    """Small prepare/sample interface with explicit n0 versus periodic geometry.

    n0 uses the angular mean of the CLIPPED TOTAL source, and uniform dummy phi.
    That preserves the native RZ marginal, up to angular/table discretization,
    not the native joint phi law or legacy angle-stream replay.
    periodic_phi retains nearest-periodic-angle Gaussians and clips in 3D;
    source phi sampling is valid even for H=1 fields; higher harmonics remain
    outside the supported coupled field solver.
    """
    def __init__(self, *, geometry, n_tor, cells_per_width=8, phi_cells=None,
                 max_cells=2_000_000):
        if geometry not in ('axisymmetric_n0', 'periodic_phi'):
            raise ValueError('explicit axisymmetric_n0 or periodic_phi geometry required')
        if type(n_tor) is not int or n_tor < 1:
            raise ValueError('positive declared n_tor required')
        if type(cells_per_width) is not int or cells_per_width < 2:
            raise ValueError('at least two cells per minimum source width required')
        if type(max_cells) is not int or max_cells < 1:
            raise ValueError('positive table cell budget required')
        if geometry == 'axisymmetric_n0':
            if n_tor != 1 or phi_cells not in (None, 1):
                raise ValueError('axisymmetric birth mode requires n_tor=1 and no phi resolution')
            phi_cells = 1
        elif type(phi_cells) is not int or phi_cells < 2:
            raise ValueError('periodic mode requires explicit phi_cells>=2; source phi exists even at n_tor=1')
        self.geometry = geometry
        self.cells_per_width = cells_per_width
        self.phi_cells = phi_cells
        self.max_cells = max_cells
        self.configuration = dict(version='native-mass-clipped-total-source-cylindrical-table-v3',
            geometry=geometry, n_tor=n_tor, cells_per_width=cells_per_width,
            phi_cells=phi_cells, max_cells=max_cells,
            toroidal_policy=('mean_of_total_clip_uniform_dummy_phi' if geometry == 'axisymmetric_n0'
                             else 'nearest_periodic_angle_vertex_table'),
            angular_integration='common_phi_analytic_clipped_mean_other_phi_midpoint_min64_width_resolved',
            angular_rounding='common_phi_continuous_integral_then_real4_node_approximation',
            approximation='positive_trilinear_acceptance_R_squared_Z_phi_exact_cell_volume',
            clipping=NATIVE_BIRTH_ACCEPTANCE,
            rng='numpy_global_MT19937_four_uniforms_per_draw_not_legacy_seven',
            cache='owned_event_snapshot_never_cross_event', production_qualified=False)
        self._bound_configuration = deepcopy(self.configuration)
        self._bound_controls = (geometry, cells_per_width, phi_cells, max_cells)

    def validate_configuration(self):
        """Changing a bound algorithm needs a new sampler/checkpoint mode."""
        if ((self.geometry, self.cells_per_width, self.phi_cells, self.max_cells) != self._bound_controls
            or self.configuration != self._bound_configuration):
            raise ValueError('birth sampler configuration changed after binding')

    def prepare(self, kernel, shards, *, time_si, bounds, dv, norm):
        """Bind the same source amplitudes, source volumes and local norm as births."""
        self.validate_configuration()
        bounds = np.asarray(bounds, dtype=float)
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or bounds[0, 0] < 0 or (bounds[1] <= bounds[0]).any()):
            raise ValueError('finite cylindrical lower/upper RZ bounds required')
        if not np.isfinite([dv, norm]).all() or min(dv, norm) <= 0:
            raise ValueError('positive finite birth dv and local source norm required')
        prepared = kernel.prepare_rates(shards, time_si=time_si)
        widths = kernel.widths(shards)
        live = (np.asarray(shards['radius']) > 0) & (np.asarray(shards['ablation_atoms_s']) > 0)
        if not live.any() or not prepared.blocks:
            raise ValueError('birth position distribution has zero active source')
        width = float(np.min(widths[live]))
        def cell_count(length):
            count = float(length)/width*self.cells_per_width
            # Remove only arithmetic roundoff at an integer cell count; retain
            # deterministic nested grids when resolution doubles.
            return max(1, ceil(count-16*np.finfo(float).eps*max(1., count)))
        nr, nz = (cell_count(length) for length in bounds[1]-bounds[0])
        shape = (nr, nz, self.phi_cells)
        if nr*nz*self.phi_cells > self.max_cells:
            raise ValueError('birth position table cell budget exceeded; no silent coarsening')
        r_edges = np.linspace(bounds[0, 0], bounds[1, 0], nr+1)
        z_edges = np.linspace(bounds[0, 1], bounds[1, 1], nz+1)
        phi_edges = np.linspace(0., 2*pi, self.phi_cells+1)
        r=r_edges;z=z_edges
        # Cache separable spatial bases once, then sum all shard contributions
        # with one matrix product per block/angle. Do NOT allocate a whole RZ
        # array for every distinct shard phi (real clouds have many centers).
        bases=[];centers_all=[]
        for positions, width_squared, amplitude, _ in prepared.blocks:
            pr = np.exp(-(r[None, :]-positions[:, 0, None])**2/width_squared)
            pz = np.exp(-(z[None, :]-positions[:, 1, None])**2/width_squared)
            centers=positions[:,2]%(2*pi)
            bases.append((pr,pz,amplitude,centers));centers_all.append(centers)
        centers_all=np.concatenate(centers_all);unique_centers=np.unique(centers_all)
        def rates_at_angle(angle):
            rate=np.zeros((nr+1,nz+1))
            for pr,pz,amplitude,centers in bases:
                distance=np.abs((angle-centers+pi)%(2*pi)-pi)
                rate+=pr.T@((amplitude*np.exp(-(distance/prepared.deltaphi)**2))[:,None]*pz)
            return rate
        nodes=np.zeros((nr+1,nz+1,self.phi_cells+1));maximum=0.;overflows=0
        angular_points=0
        if self.geometry=='axisymmetric_n0' and len(unique_centers)==1:
            common_rate=rates_at_angle(float(unique_centers[0]));peak=common_rate*dv/norm
            # Exact continuum mean of min(peak*exp(-theta^2/delta^2),1).
            # NOT min(mean(source),1). Real*4 angular rounding is approximated
            # by rounding the mean node; this is not pointwise real*4 replay.
            averaged=peak*prepared.deltaphi*sqrt(pi)*erf(pi/prepared.deltaphi)/(2*pi)
            edge=prepared.deltaphi*np.sqrt(np.log(np.maximum(peak,1.)))
            saturated=(peak>1)&(edge<pi)
            averaged[edge>=pi]=1.
            t=edge[saturated]
            averaged[saturated]=(t+peak[saturated]*prepared.deltaphi*sqrt(pi)/2*
                (erf(pi/prepared.deltaphi)-array_erf(t/prepared.deltaphi)))/pi
            nodes[:,:,0]=np.clip(averaged.astype(np.float32).astype(float),0.,1.)
            acceptance=native_birth_threshold(common_rate,dv=dv,norm=norm)
            maximum=float(np.max(acceptance));overflows=int(np.count_nonzero(~np.isfinite(acceptance)))
        else:
            if self.geometry=='axisymmetric_n0':
                angular_points=max(64,ceil(2*pi/prepared.deltaphi*self.cells_per_width))
                if angular_points>4096:raise ValueError('n0 angular integration budget exceeded; use resolved periodic mode')
                anchor=centers_all[0]
                angles=anchor+2*pi*(np.arange(angular_points)+.5)/angular_points
            else:angles=phi_edges[:-1]
            for ip,angle in enumerate(angles):
                rate=rates_at_angle(angle)
                acceptance=native_birth_threshold(rate,dv=dv,norm=norm)
                maximum=max(maximum,float(np.max(acceptance)))
                overflows+=int(np.count_nonzero(~np.isfinite(acceptance)))
                if self.geometry=='axisymmetric_n0':nodes[:,:,0]+=np.minimum(acceptance,1.)/angular_points
                else:nodes[:,:,ip]=np.minimum(acceptance,1.)
            nodes[:,:,0]=np.minimum(nodes[:,:,0],1.)
        nodes[:,:,-1]=nodes[:,:,0]  # exact periodic closure, or equal n0 faces
        volume = (.5*np.diff(r_edges**2))[:, None, None]*np.diff(z_edges)[None, :, None]*np.diff(phi_edges)[None, None, :]
        average=sum(nodes[i:i+nr,j:j+nz,k:k+self.phi_cells]
                    for i in (0,1) for j in (0,1) for k in (0,1))/8
        fully_saturated=average==1.
        mass = average*volume
        total = float(np.sum(mass))
        if not np.isfinite(total) or total <= 0:
            raise ValueError('birth position distribution has zero or invalid integral')
        probabilities = mass/total
        cdf = np.cumsum(probabilities.ravel())
        # Re-normalize cumulative roundoff; keep zero-mass cells flat.
        cdf /= cdf[-1]
        cdf[-1] = 1.
        diagnostics = dict(shape=list(shape), table_cells=int(mass.size),
            minimum_live_width_m=width,
            max_acceptance_before_clip=(maximum if np.isfinite(maximum) else None),
            float32_overflow_cells=overflows,source_phi_groups=len(unique_centers),angular_points=angular_points,
            saturated_nodes=int(np.count_nonzero(nodes>=1)),
            saturated_cells=int(np.count_nonzero(fully_saturated)),
            saturated_probability=float(np.sum(probabilities[fully_saturated])),
            acceptance_integral_m3=total,
            mean_proposal_acceptance=total/float(np.sum(volume)),
            continuous_source_qualified=False, mesh_location_included=False)
        arrays = [r_edges, z_edges, phi_edges, probabilities, cdf, nodes]
        diagnostics['owned_table_bytes'] = sum(a.nbytes for a in arrays)
        for a in arrays:
            a.flags.writeable = False
        return PreparedBirthPositions(*arrays, diagnostics)
