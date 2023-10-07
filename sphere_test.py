import colorcet as cc
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import openmesh
import pyvista as pv
import pyvistaqt as pvqt
import scipy
import time
import trimesh

from cached_property import cached_property

############################################################################
# Simulation parameters:

num_verts = 1000

############################################################################
# Common functions and classes:

# Convenience functions:
norm = np.linalg.norm
normalized = lambda x: x/norm(x)

class PolyCurve:
    def __init__(self, P):
        self._P = P

    def __call__(self, t):
        return np.array([x(t) for x in self._P]).T

    @property
    def degree(self):
        p = self._P[0]
        assert len(self._P) == 1 or all(_.order for _ in self._P[1:])
        return p.order

    @cached_property
    def D(self):
        return PolyCurve([p.deriv() for p in self._P])

    @cached_property
    def L(self):
        # Approximate arc length using Simpson's rule: should do
        # something smarter eventually but this is OK for now.
        assert self.degree <= 3
        h = self.h
        return 2*h*(norm(self.D(0)) + 4*norm(self.D(h/2)) + norm(self.D(h)))/3

class LagrangeCurve(PolyCurve):
    def __init__(self, X):
        self.X = X # Interpolated points
        self.n = X.shape[0] # Number of points
        self.d = X.shape[1] # Ambient dimension

        # Solve Lagrange interpolation problem for each component and
        # initialize parent with these polynomials:
        self.h = norm(X[-1] - X[0])
        self.T = np.linspace(0, self.h, self.n)
        P = [scipy.interpolate.lagrange(self.T, self.X[:, i]) for i in range(self.d)]
        super().__init__(P)

def get_fibonacci_sphere_points(n):
    x = np.linspace(1, -1, n)
    r, theta = np.sqrt(1 - x**2), np.pi*(np.sqrt(5) - 1)*np.arange(n)
    return np.array([x, r*np.cos(theta), r*np.sin(theta)]).T

def project_into_TP(x, v):
    return v - (x@v)*x

def d_geo(x,  y):
    '''The geodesic distance between two points on the sphere.'''
    return np.arccos(x@y)

def grad_d_geo(x, y, i=0):
    '''The surface gradient of the geodesic distance with respect to
    the `i`th component.

    '''
    if i == 0:
        return project_into_TP(x, -y/np.sqrt(1 - (x@y)**2))
    elif i == 1:
        return project_into_TP(y, -x/np.sqrt(1 - (x@y)**2))
    else:
        raise ValueError('i should be 0 or 1')

############################################################################
# A simple test:

# Sample two points on the unit sphere which subtend and arc no
# greater than pi/4 radians:
while True:
    x0, x1 = (normalized(_) for _ in np.random.randn(2, 3))
    theta = np.arccos(x0@x1)
    if theta < np.pi/4:
        break

# Compute the midpoint of the spherical arc connecting x0 and x1:
xm = normalized((x0 + x1)/2)

X = np.array([x0, xm, x1])
phi = LagrangeCurve(X)

# Use Simpson's rule (1st Gauss-Lobatto rule) to compute the arc length:
L = phi.h*(norm(phi.D(0)) + 4*norm(phi.D(phi.h/2)) + norm(phi.D(phi.h)))/6
L_gt = theta
L_rel_error = abs(L - L_gt)/abs(L)

# Check the pointwise error of our curve approximation:
T = np.linspace(0, phi.h, 1001)
Phi = phi(T)
PhiNormalized = Phi/np.sqrt(np.sum(Phi**2, axis=1)).reshape(-1, 1)
Error = np.sqrt(np.sum((Phi - PhiNormalized)**2, axis=1))
Error[Error < np.finfo(np.float64).resolution] = np.nan
# plt.figure()
# plt.semilogy(T, Error, zorder=1, linewidth=2, c='r', linestyle='-')
# plt.xlim(0, phi.h)
# for p in [0, 1, 2, 3, 4, 5]:
#     plt.axhline(y=phi.h**p, zorder=0, linewidth=1, linestyle='--', c='k')
# plt.axhline(y=L_rel_error, linewidth=1, c='b', linestyle='--', zorder=1)
# plt.show()

############################################################################
# Build a mesh on the sphere:

X = get_fibonacci_sphere_points(num_verts)
F = scipy.spatial.ConvexHull(X).simplices # facets of convex hull of
                                          # points on sphere ==
                                          # spherical Delaunay
                                          # triangulation
tm = trimesh.Trimesh(X, F)
assert tm.is_watertight

############################################################################
# Set up the marcher

FAR, TRIAL, VALID = 0, 1, 2 # Marcher states
NOT_IN_HEAP, FINALIZED = -1, -2 # Special values in `HeapIndex`
NO_INDEX = -1 # Special value indicating that the heap is unoccupied
              # at this position

Dist = np.empty(num_verts)
Dist[:] = np.inf

GradDist = np.empty((num_verts, 3))
GradDist[:, :] = np.nan

State = np.empty(num_verts, dtype=np.int8)
State[:] = FAR

# Mapping from node index to heap index:
HeapIndex = np.empty(num_verts, dtype=int)
HeapIndex[:] = NOT_IN_HEAP

# The heap backing the marcher (contains node indices):
Heap = []

def heap_swap(p, q):
    '''Swap nodes i and j in the heap, taking care to update both Heap and HeapIndex.'''
    i, j = Heap[p], Heap[q]
    Heap[p], Heap[q] = Heap[q], Heap[p]
    HeapIndex[i], HeapIndex[j] = HeapIndex[j], HeapIndex[i]

def heap_parent(p):
    assert p > 0
    return (p - 1)//2

def heap_children(p):
    return tuple(q for q in (2*p + 1, 2*p + 2) if q < len(Heap))

def heap_value(p):
    return Dist[Heap[p]]

def heap_swim(p):
    while p > 0 and heap_value(p) < heap_value(heap_parent(p)):
        q = heap_parent(p)
        heap_swap(p, q)
        p = q

def heap_insert(i):
    assert State[i] == TRIAL

    # TODO: there's some weird bug happening here. Need to fix.
    if HeapIndex[i] != NOT_IN_HEAP:
        return False

    HeapIndex[i] = len(Heap)
    Heap.append(i)

    p = HeapIndex[i]
    heap_swim(p)

    return True

def heap_min_value():
    assert Heap
    return heap_value(0)

def heap_pop():
    assert Heap
    i = Heap[0]

    p = 0
    children = heap_children(p)
    while children:
        q = children[np.argmin([heap_value(q) for q in children])]
        heap_swap(p, q)
        p = q
        children = heap_children(p)
    Heap[p] = NO_INDEX

    return i

def heap_OK():
    return all(heap_value(heap_parent(p)) <= heap_value(p) for p in range(1, len(Heap)))

def vv(i):
    return [j for j in tm.vertex_neighbors[i] if j >= 0]

def vf(i):
    return [j for j in tm.vertex_faces[i] if j >= 0]

def approx_geodesic_with_cubic(x, y):
    theta = np.arccos(x@y)
    q1 = x/norm(x)
    n = np.cross(q1, y)
    n /= norm(n)
    q2 = np.cross(n, q1)
    q2 /= norm(q2)
    X = np.array([
        x,
        np.cos(theta/3)*q1 + np.sin(theta/3)*q2,
        np.cos(2*theta/3)*q1 + np.sin(2*theta/3)*q2,
        y
    ])
    return LagrangeCurve(X)

############################################################################
# Set up and run the simulation:

eps = 1e-7

# Factoring radius
rfac = 0.2

# Initial point
i_init = 0
Dist[i_init] = 0
State[i_init] = VALID

# Compute the exact geodesic distance and surface gradient at each
# point (latter not well-defined at source or antipodal point):
DistExact = np.array([d_geo(X[i_init], x) for x in X])
GradDistExact = np.array([normalized(grad_d_geo(X[i_init], x, i=1)) for x in X])

# Exactly initialize inside the ball near the source point:
mask = DistExact <= rfac
print(f'number of points in initialization region: {mask.sum()}')
Dist[mask] = DistExact[mask]
GradDist[mask] = GradDistExact[mask]
State[mask] = VALID
HeapIndex[mask] = FINALIZED

# We now have a layer of exactly initialized VALID points with all
# remaining nodes FAR with a value of infinity. We want to set all of
# the FAR neighbors the VALID guys to TRIAL and insert them into the
# heap with a correctly initialized value.
for i in np.where(mask)[0]:
    for j in vv(i):
        if State[j] == FAR:
            Dist[j] = DistExact[j]
            GradDist[j] = normalized(GradDistExact[j])
            State[j] = TRIAL
            heap_insert(j)

start_time = time.time()

# Start marching:
while Heap:
    assert heap_OK()

    # Pop the first node from the heap, make sure it has a TRIAL state
    # and that its value is finite, and set its state to VALID,
    # finalizing it.
    i0 = heap_pop()
    assert State[i0] == TRIAL
    assert np.isfinite(Dist[i0])
    State[i0] = VALID

    # Insert all of the newly VALID nodes FAR neighbors into the heap,
    # after marking their state TRIAL.
    #
    # TODO: heap_insert can fail near the antipodal point for some
    # reason related to heap bookkeeping. Need to fix.
    should_terminate = False
    for i in vv(i0):
        if State[i] == FAR:
            State[i] = TRIAL
            if not heap_insert(i):
                should_terminate = True
                break
    if should_terminate:
        break

    # A convenience function which will generate each update index i1
    # which we should consider below:
    def opposite_inds_for_update():
        for I in tm.faces[vf(i0)]:
            mask = State[I] == VALID
            num_valid = sum(mask)
            if num_valid == 2:
                _ = np.setdiff1d(I[mask], i0)
                assert(len(_) == 1)
                yield _[0]

    # For each pair of indices (i, i1), we do the triangle update (i, i0, i1):
    for i1 in opposite_inds_for_update():
        for i in [_ for _ in vv(i0) if State[_] == TRIAL]:
            # Get useful point and tangent data at update points:
            x0, x1, x = X[i0], X[i1], X[i]
            g0, g1 = GradDist[i0], GradDist[i1]

            # Approximate the geodesic connecting the two points on
            # the base of the update triangle, x0 and x1, with a cubic
            # spline with a chordal parametrization (more generally,
            # solve the two-point BVP over a short enough distance
            # such that we can hope for the solution's uniqueness):
            x01 = approx_geodesic_with_cubic(x0, x1)

            # TODO: an optimization. If we check the angles the
            # updating geodesic can make with the interpolated
            # gradient at the base of the update, we can reject an
            # update this way. (That is, we can check at this point
            # whether it is even possible for a minimizer to be
            # oriented correctly, or whether we should just skip this
            # update.)

            # Get data needed to approximate the geodesic distance
            # field over the base of the triangle update:
            d0, d1 = Dist[i0], Dist[i1]
            t0, t1 = x01.D(0), x01.D(x01.h)
            t0 /= norm(t0)
            t1 /= norm(t1)

            # Compute cubic polynomial (the function "D" below)
            # approximating geodesic distance over update base in the
            # Bernstein basis:
            b = np.array([d0, d0 + x01.h*g0@t0/3, d1 - x01.h*g1@t1/3, d1])
            def D(h):
                lam0, lam1 = (x01.h - h)/x01.h, h/x01.h
                tmp0 = lam0*b[0] + lam1*b[1]
                tmp1 = lam0*b[1] + lam1*b[2]
                tmp2 = lam0*b[2] + lam1*b[3]
                tmp0 = lam0*tmp0 + lam1*tmp1
                tmp1 = lam0*tmp1 + lam1*tmp2
                tmp0 = lam0*tmp0 + lam1*tmp1
                return tmp0

            # Cost function for update: parameter "h" is the parameter
            # of the approximate geodesic arc connecting x0 and x1,
            # and the cost is the approximated geodesic distance so
            # far ("D") plus the geodesic distance to go ("d_geo").
            F = lambda h: D(h) + d_geo(x01(h), x)

            # Take a rather brute force approach to solving the
            # minimization problem: approximate F with a Chebyshev
            # polynomial up to sufficient precision, compute the
            # derivative, find the root on the interval. The
            # minimizing argument is h_opt.
            domain = [0, x01.h]
            deg = 8
            F_cheb = np.polynomial.Chebyshev.interpolate(F, 2*deg + 1, domain)
            rel_error = abs(F_cheb.coef[(deg + 1):]).sum()
            while rel_error > eps:
                deg *= 2
                F_cheb = np.polynomial.Chebyshev.interpolate(F, 2*deg + 1, domain)
                rel_error = abs(F_cheb.coef[(deg + 1):]).sum()
            F_cheb = F_cheb.trim(eps)
            h_roots = [np.real(_) for _ in F_cheb.deriv().roots() \
                     if abs(np.imag(_)) <= eps and domain[0] <= np.real(_) <= domain[1]]
            if not h_roots:
                h_opt = [0, x01.h][np.argmin([F_cheb(domain[0]), F_cheb(domain[1])])]
            elif len(h_roots) == 1:
                h_opt = h_roots[0]
            else:
                assert False

            # Get the minimizing point on the update base.
            x_opt = x01(h_opt)

            # If there was no improvement in the geodesic distance, we
            # move on to the next update.
            if F(h_opt) > Dist[i]:
                continue

            # We also check the Lagrange multiplier for the
            # minimization problem if h_opt lies on the boundary of
            # the interval [0, x01.h]. If the Lagrange multiplier is
            # nonzero, some spurious diffraction may have occured, so
            # we reject the update (since we don't have enough
            # information to accept it at this point).
            #
            # TODO: It may be necessary to cache this update to "match
            # it" with an incident update during a later update phase.
            if h_opt in {0, x01.h}:
                dF_opt = normalized(x01.D(h_opt))@GradDist[i0 if h_opt == 0 else i1]
                if (h_opt == 0 and dF_opt > 0) or (h_opt == x01.h and dF_opt < 0):
                    continue

            # Approximate the geodesic arc connecting the optimum
            # point and the update point. We need this so that we can
            # get the surface gradient (i.e., the tangent vector of
            # this gradient) at the update point.
            xup = approx_geodesic_with_cubic(x_opt, x)

            # Set the new values:
            Dist[i] = F(h_opt)
            GradDist[i] = normalized(xup.D(xup.h))

            # Update the heap:
            heap_swim(HeapIndex[i])

finish_time = time.time()

print(f'finished in {finish_time - start_time:0.1f}s ({X.shape[0]} points)')

############################################################################
# Make a nice 3D plot showing what's going on. Note: this will show
# the last update, so it can be useful for debugging.

points = pv.PolyData(X)
points['State'] = State.astype(np.float64)
points['State'][i_init] = 3

Dist[~np.isfinite(Dist)] = np.nan
points['Dist'] = Dist

RelError = abs(DistExact - Dist)/abs(DistExact)
points['RelError'] = RelError

vmax = np.nanmax(abs(RelError))
vmin = -vmax
clim = (vmin, vmax)

plotter = pvqt.BackgroundPlotter()
# plotter.add_mesh(points.glyph(orient=False, scale=False, geom=pv.Sphere(0.02)), scalars='Dist', cmap=cc.cm.gouldian, nan_opacity=0)
plotter.add_mesh(points.glyph(orient=False, scale=False, geom=pv.Sphere(0.02)), scalars='RelError', clim=clim, cmap=cc.cm.coolwarm, nan_opacity=0)
plotter.add_mesh(pv.make_tri_mesh(tm.vertices, tm.faces), show_edges=True, color='white', opacity=1)

# Plot most recent update
plotter.add_mesh(pv.Spline(x01(np.linspace(0, x01.h))), line_width=3, color='red')
plotter.add_mesh(pv.Sphere(0.011, tm.vertices[i]), color='chartreuse')
plotter.add_mesh(pv.Arrow(x0, g0, scale=5e-2), color='red')
plotter.add_mesh(pv.Arrow(x1, g1, scale=5e-2), color='red')
plotter.add_mesh(pv.Sphere(0.011, x0), color='orange')
plotter.add_mesh(pv.Sphere(0.011, x1), color='pink')
plotter.add_mesh(pv.Spline(xup(np.linspace(0, xup.h))), line_width=3, color='red')

############################################################################
# Make a 2D plot of the relative error:

Az = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2*np.pi)
El = np.arccos(X[:, 2])

plt.figure(figsize=(8, 4))
plt.scatter(Az, El, s=30, c=RelError, cmap=cc.cm.coolwarm, vmin=0, vmax=vmax)
plt.colorbar()
plt.xlim(0, 2*np.pi)
plt.ylim(np.pi, 0)
plt.xticks(np.linspace(0, 2*np.pi, 5))
plt.gca().set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.yticks(np.linspace(0, np.pi, 3)[::-1])
plt.gca().set_yticklabels(['0', r'$\pi/2$', r'$\pi$'][::-1])
plt.xlabel(r'$\phi$ [Azimuth]')
plt.ylabel(r'$\theta$ [Elevation]')
plt.title(r'Pointwise relative error: $|\hat{d} - d|/|d|$')
plt.tight_layout()
plt.show()
