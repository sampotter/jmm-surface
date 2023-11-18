import colorcet as cc
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import skimage
import trimesh

norm = np.linalg.norm

R = 1
r = 0.2

@np.vectorize
def phi(x, y, z):
    return (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4*R**2*(x**2 + y**2)

@np.vectorize
def grad_phi(x, y, z):
    phi_x = -8*R**2*x + 4*x*(R**2 - r**2 + x**2 + y**2 + z**2)
    phi_y = -8*R**2*y + 4*y*(R**2 - r**2 + x**2 + y**2 + z**2)
    phi_z = 4*z*(R**2 - r**2 + x**2 + y**2 + z**2)
    return np.array([phi_x, phi_y, phi_z])

def project(p, h, eps=1e-15):
    if abs(phi(*p)) <= eps:
        return p
    g = grad_phi(*p)
    g /= norm(g)
    f = np.vectorize(lambda t: phi(*(p - t*g)))
    tmax = h/np.sqrt(3)
    F = np.polynomial.Chebyshev.interpolate(f, 16, [-tmax, tmax])
    roots = [np.real(z) for z in F.roots()
             if abs(np.imag(z)) <= 1e-15
             and -tmax <= np.real(z) <= tmax]
    if len(roots) != 1:
        import ipdb; ipdb.set_trace()
    t = roots[0]
    return p - t*g

if __name__ == '__main__':
    n = 101
    _ = np.linspace(-2, 2, n)
    h = _[1] - _[0]
    x, y, z = np.meshgrid(_, _, _, indexing='xy')
    del _

    verts, faces, _, values = \
        skimage.measure.marching_cubes(phi(x, y, z), level=0, spacing=(h, h, h))
    del _

    verts += np.array([x.min(), y.min(), z.min()])

    for i, vert in enumerate(verts):
        print(f'{i+1}/{verts.shape[0]}')
        verts[i] = project(vert, h, eps=1e-15)

    grid = pv.make_tri_mesh(verts, faces)
    # grid['phi'] = values
    grid['phi'] = np.array([phi(*vert) for vert in verts])

    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(grid, show_edges=True, cmap=cc.cm.gouldian)
    plotter.add_mesh(pv.Box((-2, 2, -2, 2, -2, 2)), style='wireframe')
