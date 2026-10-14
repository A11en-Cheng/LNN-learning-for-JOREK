"""Compile actual JOREK code in a temporary directory; independent cubic inputs.

No production locator imports. JOREK_ROOT may point at another checkout.
The global search is an observable, controlled stub: only nearby semantics
are compared, never numerical parity with NativeGlobal/find_RZ.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


JOREK_ROOT = Path(os.environ.get('JOREK_ROOT', '/home/ac_desktop/JOREK_code/jorek'))


@dataclass
class OracleResult:
    element: int
    flag: int
    fallback_calls: int
    st: np.ndarray
    target: np.ndarray


def compile_oracle(directory):
    """Extract whole actual routines verbatim, retaining their source semantics."""
    compiler = shutil.which('gfortran')
    if compiler is None:
        raise FileNotFoundError('gfortran is required for the nearby oracle')
    directory = Path(directory)
    reference = Path(__file__).with_name('native_nearby_reference.f90').read_text()
    modules, driver = reference.split('program native_nearby_reference\n', 1)
    neighbor_source = (JOREK_ROOT / 'grids/mod_neighbours.f90').read_text()
    routines = []
    for name in ('neighbours_side_co_counter', 'coord_in_neighbour'):
        match = re.search(rf'^subroutine {name}\b.*?^end subroutine {name}\b',
                          neighbor_source, re.M | re.S | re.I)
        if match is None:
            raise ValueError(f'actual JOREK routine {name} missing')
        routines.append(match.group())
    generated = {
        'stubs.f90': modules,
        'neighbours.f90': 'module mod_neighbours\nuse data_structure\nimplicit none\ncontains\n'
                          + '\n'.join(routines) + '\nend module\n',
        'driver.f90': 'program native_nearby_reference\n' + driver,
    }
    for name, content in generated.items():
        (directory / name).write_text(content)
    actual = JOREK_ROOT / 'grids/grid_utils/mod_find_rz_nearby.f90'
    executable = directory / 'nearby_oracle'
    subprocess.run([compiler, '-cpp', '-O0', '-fcheck=all', '-ffree-line-length-none',
                    '-DSTELLARATOR_MODEL=0', str(directory / 'stubs.f90'),
                    str(directory / 'neighbours.f90'), str(actual),
                    str(directory / 'driver.f90'), '-o', str(executable)],
                   cwd=directory, check=True, capture_output=True, text=True)
    return executable


def run_oracle(executable, polynomial, vertex, neighbours, old, st, target,
               *, mode=0, fallback=(0, 99, .25, .75)):
    n = polynomial.shape[0]
    lines = [f'{mode} {n}']
    for e in range(n):
        lines.append(' '.join(map(str, vertex[:, e])))
        lines.append(' '.join(map(str, neighbours[:, e])))
        for j in range(4):
            for i in range(4):
                lines.append(' '.join(format(v, '.17g') for v in polynomial[e, :, i, j]))
    lines.append(' '.join(map(str, [old, *st, *target])))
    lines.append(' '.join(map(str, fallback)))
    result = subprocess.run([str(executable)], input='\n'.join(lines)+'\n',
                            check=True, capture_output=True, text=True)
    # Broken reciprocal topology writes a diagnostic before the numeric record.
    fields = result.stdout.splitlines()[-1].split()
    return OracleResult(*(int(v) for v in fields[:3]),
                        np.array(fields[3:5], float), np.array(fields[5:7], float))


def geometry(polynomial, e, st):
    """Analytic tensor monomials, independent of native/JOREK basis functions."""
    s, t = st
    x = np.zeros(2); jac = np.zeros((2, 2))
    for j in range(4):
        for i in range(4):
            c = polynomial[e, :, i, j]
            x += c*s**i*t**j
            if i:
                jac[:, 0] += c*i*s**(i-1)*t**j
            if j:
                jac[:, 1] += c*j*s**i*t**(j-1)
    return x, jac


def grid(cells=3, *, curved=False, reversed_edge=False):
    """Conforming strip X=e+s, Y=t; reversed cell uses X=e+1-s,Y=1-t."""
    p = np.zeros((cells, 2, 4, 4))
    vertex = np.array([[2*e+1, 2*e+3, 2*e+4, 2*e+2] for e in range(cells)]).T
    nb = np.zeros((4, cells), dtype=int)
    for e in range(cells):
        p[e, 0, 0, 0] = 2+e; p[e, 0, 1, 0] = 1
        p[e, 1, 0, 1] = 1
        if curved:
            p[e, 0, 0, 1] = .1; p[e, 0, 0, 2] = -.1
            p[e, 1, 0, 1] += .12*e; p[e, 1, 0, 2] = -.12*e
            p[e, 1, 1, 1] = .12; p[e, 1, 1, 2] = -.12
        if e+1 < cells:
            nb[1, e] = e+2
        if e:
            nb[3, e] = e
    if reversed_edge:
        # Exact polynomial substitution s->1-s,t->1-t.
        from math import comb
        old = p[1].copy(); p[1] = 0
        for i in range(4):
            for j in range(4):
                for a in range(i+1):
                    for b in range(j+1):
                        p[1, :, a, b] += old[:, i, j]*comb(i, a)*comb(j, b)*(-1)**(a+b)
        vertex[:, 1] = vertex[[2, 3, 0, 1], 1]
        nb[:, 1] = nb[[2, 3, 0, 1], 1]
    return p, vertex, nb


def native_mesh(polynomial, vertex):
    """Convert analytic endpoint jets to a genuine NativeSourceMesh.

    Shared vertex IDs match Fortran; size signs encode reversed local jets.
    """
    from src.models.native_coupling import NativeSourceMesh
    from src.models.native_state import NativeUnits
    n = len(polynomial)
    nodes = int(vertex.max())
    coordinates = np.zeros((2, 4, 1, nodes))
    seen = set()
    size = np.repeat(np.array([[1, 1, 1, 1], [1, -1, -1, 1],
                              [1, 1, -1, -1], [1, -1, 1, -1]], dtype=float)[:, :, None], n, axis=2)
    for e in range(n):
        for v, st in enumerate(((0., 0.), (1., 0.), (1., 1.), (0., 1.))):
            x, jac = geometry(polynomial, e, st)
            mixed = sum((polynomial[e, :, i, j]*i*j*st[0]**(i-1)*st[1]**(j-1)
                         for i in range(1, 4) for j in range(1, 4)), np.zeros(2))
            node = int(vertex[v, e])-1
            jets = np.column_stack((x, jac[:, 0]/3, jac[:, 1]/3, mixed/9))
            if node not in seen:
                coordinates[:, :, 0, node] = jets
                seen.add(node)
            else:
                stored = coordinates[:, :, 0, node]
                for degree in range(1, 4):
                    component = int(np.argmax(np.abs(stored[:, degree])))
                    ratio = (jets[component, degree]/stored[component, degree]
                             if stored[component, degree] != 0 else 1.)
                    np.testing.assert_allclose(jets[:, degree], stored[:, degree]*ratio,
                                               rtol=0, atol=1e-12)
                    size[degree, v, e] *= ratio
    return NativeSourceMesh(coordinates, size, vertex-1,
                            np.arange(1, 4*nodes+1).reshape(4, nodes),
                            units=NativeUnits(1., 1., 1., 5/3), projection='none')
