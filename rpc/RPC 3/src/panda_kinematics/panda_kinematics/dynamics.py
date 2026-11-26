"""Seven-coordinate Panda dynamics derived from rigid-body energies."""

import numpy as np


def _vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must contain seven finite values')
    return values


class PandaDynamics:
    """Evaluate Lagrangian terms using static geometry and batched NumPy."""

    def __init__(self, model):
        self.model = model
        self._mass = np.array([body.mass for body in model.bodies])
        self._inertia = np.array([body.inertia for body in model.bodies])
        self._local_com = np.array([body.com for body in model.bodies])
        self._armature = np.diag(model.armature)
        joints = [body for body in model.bodies if body.arm_joint is not None]
        axes = np.array([body.axis for body in joints])
        self._joint_reference = np.array([body.reference for body in joints])
        self._skew = np.cross(axes[:, None, :], np.eye(3)).swapaxes(-1, -2)
        self._skew_squared = self._skew @ self._skew
        self._ancestors = np.zeros((len(model.bodies), 7), dtype=bool)
        for i, body in enumerate(model.bodies):
            if body.parent >= 0:
                self._ancestors[i] = self._ancestors[body.parent]
            if body.arm_joint is not None:
                self._ancestors[i, body.arm_joint] = True

    def _kinematics(self, q):
        """Evaluate nominal and perturbed poses in the same traversal."""
        count = len(q)
        nbody = len(self.model.bodies)
        positions = np.empty((count, nbody, 3))
        rotations = np.empty((count, nbody, 3, 3))
        origins = np.empty((count, 7, 3))
        axes = np.empty_like(origins)
        angle = (q - self._joint_reference)[..., None, None]
        hinge = np.eye(3) + np.sin(angle)*self._skew + (1 - np.cos(angle))*self._skew_squared
        for i, body in enumerate(self.model.bodies):
            if body.parent < 0:
                position, orientation = body.offset, body.rotation
            else:
                parent = rotations[:, body.parent]
                position = positions[:, body.parent] + parent @ body.offset
                orientation = parent @ body.rotation
            joint = body.arm_joint
            if joint is not None:
                origins[:, joint] = position + orientation @ body.anchor
                axes[:, joint] = orientation @ body.axis
                orientation = orientation @ hinge[:, joint]
                position = origins[:, joint] - orientation @ body.anchor
            positions[:, i], rotations[:, i] = position, orientation

        com = positions + (rotations @ self._local_com[..., None])[..., 0]
        offset = com[:, :, None, :] - origins[:, None, :, :]
        joint_axes = axes[:, None, :, :] * self._ancestors[None, :, :, None]
        linear = np.cross(joint_axes, offset).swapaxes(-1, -2)
        return com, rotations, linear, joint_axes.swapaxes(-1, -2)

    def com_kinematics(self, q):
        """Return each body's world COM, orientation, and COM Jacobians."""
        return list(zip(*(array[0] for array in self._kinematics(_vector(q, 'q')[None]))))

    def _mass_matrix(self, inertia, jv, jw):
        linear = jv.swapaxes(-1, -2) @ jv
        angular = jw.swapaxes(-1, -2) @ inertia @ jw
        return self._armature + (self._mass[:, None, None]*linear + angular).sum(axis=0)

    def mass_matrix(self, q):
        """Return the kinetic-energy Hessian with respect to joint velocity."""
        _, rotation, jv, jw = self._kinematics(_vector(q, 'q')[None])
        inertia = rotation[0] @ self._inertia @ rotation[0].swapaxes(-1, -2)
        return self._mass_matrix(inertia, jv[0], jw[0])

    def potential_energy(self, q):
        """Return gravitational potential energy in joules."""
        com, _, _, _ = self._kinematics(_vector(q, 'q')[None])
        return -self._mass @ (com[0] @ self.model.gravity)

    def _gravity(self, jv):
        return -(self._mass[:, None] * (jv.swapaxes(-1, -2) @ self.model.gravity)).sum(axis=0)

    def gravity(self, q):
        """Return the seven-component potential-energy gradient."""
        _, _, jv, _ = self._kinematics(_vector(q, 'q')[None])
        return self._gravity(jv[0])

    def terms(self, q, dq):
        """Return M(q), C(q,dq)dq, and g(q), excluding passive forces."""
        q, dq = _vector(q, 'q'), _vector(dq, 'dq')
        moving = np.any(dq)
        step = 1e-5 / max(1.0, np.max(np.abs(dq)))
        poses = np.stack((q, q + step*dq, q - step*dq)) if moving else q[None]
        _, rotations, jv, jw = self._kinematics(poses)
        inertia = rotations[0] @ self._inertia @ rotations[0].swapaxes(-1, -2)
        mass = self._mass_matrix(inertia, jv[0], jw[0])
        coriolis = np.zeros(7)
        if moving:
            # Jdot*dq gives the accelerations at zero joint acceleration.
            linear_acc = ((jv[1] - jv[2]) @ dq) / (2*step)
            angular_acc = ((jw[1] - jw[2]) @ dq) / (2*step)
            omega = jw[0] @ dq
            moment = ((inertia @ angular_acc[..., None])[..., 0]
                      + np.cross(omega, (inertia @ omega[..., None])[..., 0]))
            force = self._mass[:, None]*linear_acc
            coriolis = ((jv[0].swapaxes(-1, -2) @ force[..., None])
                        + (jw[0].swapaxes(-1, -2) @ moment[..., None])).sum(axis=0)[:, 0]
        return mass, coriolis, self._gravity(jv[0])
