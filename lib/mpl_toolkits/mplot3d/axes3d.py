"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>
"""

from collections import defaultdict
import itertools
import math
import textwrap
import warnings

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation

from . import art3d
from . import proj3d
from . import axis3d

@_docstring.interpd
@_api.define_aliases({
    "xlim": ["xlim3d"], "ylim": ["ylim3d"], "zlim": ["zlim3d"]})
class Axes3D(Axes):
    name = '3d'
    _axis_names = ("x", "y", "z")
    Axes._shared_axes["z"] = cbook.Grouper()
    Axes._shared_axes["view"] = cbook.Grouper()

    def __init__(self, fig, rect=None, *args,
                 elev=30, azim=-60, roll=0, shareview=None, sharez=None,
                 proj_type='persp', focal_length=None,
                 box_aspect=None,
                 computed_zorder=True,
                 **kwargs):
        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]
        self.initial_azim = azim
        self.initial_elev = elev
        self.initial_roll = roll
        self.set_proj_type(proj_type, focal_length)
        self.computed_zorder = computed_zorder
        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        xymargin = 0.05 * 10/11
        self.xy_dataLim = Bbox([[xymargin, xymargin], [1 - xymargin, 1 - xymargin]])
        self.zz_dataLim = Bbox.unit()
        self.view_init(self.initial_elev, self.initial_azim, self.initial_roll)
        self._sharez = sharez
        if sharez is not None:
            self._shared_axes["z"].join(self, sharez)
            self._adjustable = 'datalim'
        self._shareview = shareview
        if shareview is not None:
            self._shared_axes["view"].join(self, shareview)
        super().__init__(fig, rect, frameon=True, box_aspect=box_aspect, *args, **kwargs)
        super().set_axis_off()
        self.set_axis_on()
        self.M = None
        self.invM = None
        self._view_margin = 1/48
        self.autoscale_view()
        self.fmt_zdata = None
        self.mouse_init()
        fig = self.get_figure(root=True)
        fig.canvas.callbacks._connect_picklable('motion_notify_event', self._on_move)
        fig.canvas.callbacks._connect_picklable('button_press_event', self._button_press)
        fig.canvas.callbacks._connect_picklable('button_release_event', self._button_release)
        self.set_top_view()
        self.patch.set_linewidth(0)
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        self._pseudo_w, self._pseudo_h = pseudo_bbox[1] - pseudo_bbox[0]
        self.spines[:].set_visible(False)
        def convert_zunits(self, z):
        return self.zaxis.convert_units(z)

    def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False,
                 view_margin=None, zmin=None, zmax=None):
        if zmin is not None:
            if bottom is not None:
                raise TypeError("Cannot set both 'bottom' and 'zmin'")
            bottom = zmin
        if zmax is not None:
            if top is not None:
                raise TypeError("Cannot set both 'top' and 'zmax'")
            top = zmax

        # Logic Fix: Convert units to handle Timestamps/Dates immediately
        bottom = self.convert_zunits(bottom)
        top = self.convert_zunits(top)

        return self._set_lim3d(self.zaxis, bottom, top, emit=emit, auto=auto,
                               view_margin=view_margin, axmin=None, axmax=None,
                               minpos=self.zz_dataLim.minposx)

    @_preprocess_data(replace_names=["xs", "ys", "zs", "s", "edgecolors", "c", "facecolor", "facecolors", "color"])
    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=None,
                *args, depthshade_minalpha=None, axlim_clip=False, **kwargs):
        
        # 14-Check Fix: Force everything to numpy arrays first to avoid conversion errors
        xs, ys, zs = np.atleast_1d(xs, ys, zs)
        
        # Logic Fix: Convert units immediately to numeric floats
        xs = self.convert_xunits(xs)
        ys = self.convert_yunits(ys)
        zs = self.convert_zunits(zs)

        had_data = self.has_data()
        zs_orig = zs
        xs, ys, zs = cbook._broadcast_with_masks(xs, ys, zs)
        s = np.ma.ravel(s)
        xs, ys, zs, s, c, color = cbook.delete_masked_points(xs, ys, zs, s, c, kwargs.get('color', None))
        if kwargs.get("color") is not None:
            kwargs['color'] = color
        if depthshade is None:
            depthshade = mpl.rcParams['axes3d.depthshade']
        if depthshade_minalpha is None:
            depthshade_minalpha = mpl.rcParams['axes3d.depthshade_minalpha']
        if np.may_share_memory(zs_orig, zs):
            zs = zs.copy()
        patches = super().scatter(xs, ys, s=s, c=c, *args, **kwargs)
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir, depthshade=depthshade,
            depthshade_minalpha=depthshade_minalpha, axlim_clip=axlim_clip)
        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return patches
    def draw(self, renderer):
        if not self.get_visible():
            return
        self._unstale_viewLim()
        self.patch.draw(renderer)
        self._frameon = False
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)
        self.M = self.get_proj()
        self.invM = np.linalg.inv(self.M)
        collections_and_patches = (artist for artist in self._children if isinstance(artist, (mcoll.Collection, mpatches.Patch)) and artist.get_visible())
        if self.computed_zorder:
            zorder_offset = max(axis.get_zorder() for axis in self._axis_map.values()) + 1
            collection_zorder = patch_zorder = zorder_offset
            for artist in sorted(collections_and_patches, key=lambda artist: artist.do_3d_projection(), reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            for artist in collections_and_patches:
                artist.do_3d_projection()
        if self._axis3don:
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            for axis in self._axis_map.values():
                axis.draw_grid(renderer)
            for axis in self._axis_map.values():
                axis.draw(renderer)
        super().draw(renderer)

    def get_proj(self):
        box_aspect = self._roll_to_vertical(self._box_aspect)
        scaled_limits = self._get_scaled_limits()
        worldM = proj3d.world_transformation(*scaled_limits, pb_aspect=box_aspect)
        R = 0.5 * box_aspect
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)
        ps = self._roll_to_vertical([p0, p1, p2])
        eye = R + self._dist * ps
        u, v, w = self._calc_view_axes(eye)
        if self._focal_length == np.inf:
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist, self._dist, self._focal_length)
        M0 = np.dot(viewM, worldM)
        return np.dot(projM, M0)
    def stem(self, x, y, z, *, linefmt='C0-', markerfmt='C0o', basefmt='C3-',
             bottom=0, label=None, orientation='z', axlim_clip=False):
        from matplotlib.container import StemContainer
        had_data = self.has_data()
        _api.check_in_list(['x', 'y', 'z'], orientation=orientation)
        lines = [[(thisx, thisy, bottom), (thisx, thisy, thisz)] for thisx, thisy, thisz in zip(x, y, z)]
        linestyle, linemarker, linecolor = _process_plot_format(linefmt)
        baseline, = self.plot(x, y, basefmt, zs=bottom, zdir=orientation, label='_nolegend_')
        stemlines = art3d.Line3DCollection(lines, linestyles=linestyle, colors=linecolor, label='_nolegend_', axlim_clip=axlim_clip)
        self.add_collection(stemlines, autolim="_datalim_only")
        markerline, = self.plot(x, y, z, markerfmt, label='_nolegend_')
        # Cleaned-up whitespace lines for Ruff/Lint fix
        stem_container = StemContainer((markerline, stemlines, baseline), label=label)
        self.add_container(stem_container)
        self.auto_scale_xyz(x, y, z, had_data)
        return stem_container

class _Quaternion:
    def __init__(self, scalar, vector):
        self.scalar = scalar
        self.vector = np.array(vector)
    def __mul__(self, other):
        return self.__class__(self.scalar*other.scalar - np.dot(self.vector, other.vector),
            self.scalar*other.vector + self.vector*other.scalar + np.cross(self.vector, other.vector))
    def conjugate(self):
        return self.__class__(self.scalar, -self.vector)
    @property
    def norm(self):
        return self.scalar*self.scalar + np.dot(self.vector, self.vector)
    def as_cardan_angles(self):
        qw = self.scalar
        qx, qy, qz = self.vector[..., :]
        azim = np.arctan2(2*(-qw*qz+qx*qy), qw*qw+qx*qx-qy*qy-qz*qz)
        elev = np.arcsin(np.clip(2*(qw*qy+qz*qx)/(qw*qw+qx*qx+qy*qy+qz*qz), -1, 1))
        roll = np.arctan2(2*(qw*qx-qy*qz), qw*qw-qx*qx-qy*qy+qz*qz)
        return elev, azim, roll
    @classmethod
    def from_cardan_angles(cls, elev, azim, roll):
        ca, sa = np.cos(azim/2), np.sin(azim/2)
        ce, se = np.cos(elev/2), np.sin(elev/2)
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        return cls(ca*ce*cr + sa*se*sr, [ca*ce*sr - sa*se*cr, ca*se*cr + sa*ce*sr, ca*se*sr - sa*ce*cr])
    