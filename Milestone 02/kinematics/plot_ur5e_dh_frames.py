"""Plot UR5e standard-DH X/Z frame assignments for Milestone 02.

Based on the validated DH table in ur5e_dh_fk.py.
Uses only x_i and z_i arrows (as in MCTR911 Tutorial 03).
At q=0; dimensions in meters. The gray line is an approximate
physical joint-center skeleton from the UR5e MJCF, not a CAD rendering.

Run in same directory as ur5e_dh_fk.py:
    python -m pip install matplotlib
    python plot_ur5e_dh_frames.py

Outputs: ur5e_dh_frames.png and ur5e_dh_frames.svg
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ur5e_dh_fk import DH_CONSTANTS, standard_dh_transform

# Physical joint centers from original ur5e.xml when q=[0]*6,
# reported in MuJoCo world coordinates (not DH frame origins).
PHYSICAL_CHAIN = np.array([
    [0.000,  0.000, 0.000],  # robot base
    [0.000,  0.000, 0.163],  # shoulder_pan
    [0.000, -0.138, 0.163],  # shoulder_lift
    [-0.425,-0.007, 0.163], # elbow
    [-0.817,-0.007, 0.163], # wrist_1
    [-0.817,-0.134, 0.163], # wrist_2
    [-0.817,-0.134, 0.063], # wrist_3
    [-0.817,-0.234, 0.063], # attachment_site
])


def frame_transforms(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError('Expected six joint values')
    frames = [np.eye(4)]
    for theta, (a, alpha, d) in zip(q, DH_CONSTANTS):
        frames.append(frames[-1] @ standard_dh_transform(theta, d, a, alpha))
    return frames


def add_arrow(ax, origin, direction, color, length, linewidth=2.5):
    direction = np.asarray(direction)
    d = length * direction / np.linalg.norm(direction)
    ax.quiver(*origin, *d, color=color, arrow_length_ratio=0.22,
              linewidth=linewidth, normalize=False)


def set_nice_axes(ax, limits, title, view):
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect((limits[0][1]-limits[0][0],
                       limits[1][1]-limits[1][0],
                       limits[2][1]-limits[2][0]), zoom=0.92)
    ax.set_xlabel('World X (m)', labelpad=10, fontsize=9)
    ax.set_ylabel('World Y (m)', labelpad=9, fontsize=9)
    ax.set_zlabel('World Z (m)', labelpad=8, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.view_init(elev=view[0], azim=view[1])
    ax.grid(True, alpha=0.15)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('#d6dee9')


def main():
    frames = frame_transforms(np.zeros(6))
    P = np.array([T[:3, 3] for T in frames])
    assert np.allclose(P[-1], [-.817, -.234, .063], atol=1e-12)
    # Only X and Z are DH-assigned. Y follows from the right-hand rule.
    X = np.array([T[:3, 0] for T in frames]); Z = np.array([T[:3, 2] for T in frames])

    plt.rcParams.update({'font.family':'DejaVu Sans','savefig.facecolor':'white'})
    fig = plt.figure(figsize=(15.8, 8.2), constrained_layout=False)
    fig.patch.set_facecolor('white')
    fig.suptitle('UR5e  |  Standard DH frame assignment at q = [0, 0, 0, 0, 0, 0]',
                 fontsize=18, fontweight='bold', y=0.965)

    ax = fig.add_axes([0.025, .19, .63, .68], projection='3d')
    set_nice_axes(ax, [(-.99, .13), (-.40, .16), (-.18, .45)],
                  'A. Full robot: DH frames {0} through {6}', (19, -66))
    ax.plot(*PHYSICAL_CHAIN.T, '-o', color='#9aa8b7', lw=5, ms=5, alpha=.65, zorder=1)
    ax.plot(*P.T, ':', color='#55697c', lw=1.3, alpha=.75)
    for i, T in enumerate(frames):
        p = P[i]
        ax.scatter(*p, color='#202e3b', s=27, zorder=8)
        add_arrow(ax, p, X[i], '#cb303b', .102, linewidth=2.2)
        add_arrow(ax, p, Z[i], '#1668c1', .102, linewidth=2.2)
        off = np.array([0., .035 if i in (0,1,2,3) else -.035,
                        .048 if i%2==0 else -.040])
        ax.text(*(p + off), f'{{{i}}}', color='#202e3b', fontsize=11, weight='bold', zorder=10)

    # Wrist close-up, where several frames are only 100 mm apart.
    ax2 = fig.add_axes([.655, .235, .32, .59], projection='3d')
    set_nice_axes(ax2, [(-.94,-.65),(-.34,.04),(-.07,.30)],
                  'B. Wrist and tool close-up', (22, -59))
    ax2.plot(*PHYSICAL_CHAIN[-5:].T, '-o', color='#9aa8b7', lw=5, ms=5, alpha=.65)
    for i in (3,4,5,6):
        p=P[i]; ax2.scatter(*p, color='#202e3b', s=30)
        add_arrow(ax2,p,X[i],'#cb303b',.065,linewidth=2.4)
        add_arrow(ax2,p,Z[i],'#1668c1',.065,linewidth=2.4)
        off = np.array([.007, -.022 if i in (3,5) else .032, .027])
        ax2.text(*(p+off), f'{{{i}}}',fontsize=10,weight='bold',color='#202e3b')

    legend = [Line2D([0],[0],color='#cb303b',lw=3,label=r'$x_i$  (assigned X axis)'),
              Line2D([0],[0],color='#1668c1',lw=3,label=r'$z_i$  (joint/endpoint Z axis)'),
              Line2D([0],[0],color='#9aa8b7',lw=5,label='Physical joint-center skeleton'),
              Line2D([0],[0],color='#55697c',lw=1.5,ls=':',label='DH frame-origin path')]
    fig.legend(handles=legend,loc='lower center', bbox_to_anchor=(.5,.125),
               frameon=False,ncol=4,fontsize=10)
    fig.text(.5,.070,
             'Frame {0} = MuJoCo world/base. Frame {6} = attachment_site.  '
             'DH frame origins need not coincide with physical joint centers.',
             ha='center',fontsize=10.5,color='#354253')
    fig.text(.5,.038,
             'Standard DH:  T(i-1,i) = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i).  '
             'Negative a2 and a3 reflect the chosen +X arrow directions.',
             ha='center',fontsize=9.5,color='#64748b')

    out=Path(__file__).resolve().parent
    for name in ('ur5e_dh_frames.png','ur5e_dh_frames.svg'):
        fig.savefig(out/name,dpi=210,bbox_inches='tight',pad_inches=.22)
        print('Created:',out/name)
    plt.close(fig)

    print('\nZero-pose DH axes (world coordinates):')
    for i in range(7):
        print(f'{{{i}}}: origin={np.round(P[i],3)} x={np.round(X[i],3)} z={np.round(Z[i],3)}')

if __name__ == '__main__':
    main()
