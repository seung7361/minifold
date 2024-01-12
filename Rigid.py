import torch
from typing import Tuple

class Rotation:
    def __init__(self, rots: torch.Tensor=None, quats: torch.Tensor=None):
        if rots is not None and quats is not None:
            pass
        elif rots is not None:
            self.rots = rots
            self.quats = Rotation.rots_to_quats(rots)
        elif quats is not None:
            self.quats = quats
            self.rots = Rotation.quats_to_rots(quats)
        else:
            raise ValueError("Either rots or quats must be specified.")
        
    def __repr__(self):
        return f"<Rotation object (rots={self.rots}, quats={self.quats})>"
    
    def rots_to_quats(rots: torch.Tensor):
        """
        Converts a batch of rotation matrices to quaternions.
        
        rots: (B, 3, 3)
        
        return: (B, 4)
        """

        rots = rots.reshape(-1, 3, 3)
        tr = rots[:, 0, 0] + rots[:, 1, 1] + rots[:, 2, 2]
        qw = torch.sqrt(1 + tr) / 2
        qx = (rots[:, 2, 1] - rots[:, 1, 2]) / (4 * qw)
        qy = (rots[:, 0, 2] - rots[:, 2, 0]) / (4 * qw)
        qz = (rots[:, 1, 0] - rots[:, 0, 1]) / (4 * qw)
        quats = torch.stack([qw, qx, qy, qz], dim=1)

        return quats

    def quats_to_rots(quats: torch.Tensor):
        """
        Converts a batch of quaternions to rotation matrices.
        
        quats: (B, 4)

        return: (B, 3, 3)
        """

        quats = quats.reshape(-1, 4)
        qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        rots = torch.stack([
            1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2
        ], dim=1).reshape(-1, 3, 3)

        return rots

    def invert_rot(rots: torch.Tensor):
        return rots.transpose(-1, -2)
    
    def invert_quat(quats: torch.Tensor):
        quats = quats.clone()
        quats[..., 1:] *= -1
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        return quats

    def mul_quats(left: torch.Tensor, right: torch.Tensor):
        """
        Multiplies two batches of quaternions.

        left: (B, 4)
        right: (B, 4)

        return: (B, 4)
        """

        left = left.reshape(-1, 4)
        right = right.reshape(-1, 4)
        lw, lx, ly, lz = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
        rw, rx, ry, rz = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
        qw = lw * rw - lx * rx - ly * ry - lz * rz
        qx = lw * rx + lx * rw + ly * rz - lz * ry
        qy = lw * ry - lx * rz + ly * rw + lz * rx
        qz = lw * rz + lx * ry - ly * rx + lz * rw
        quats = torch.stack([qw, qx, qy, qz], dim=1)

        return quats

    def mul_quats_vecs(quats: torch.Tensor, vecs: torch.Tensor):
        """
        Rotates a batch of vectors by a batch of quaternions.

        quats: (B, 4)
        vecs: (B, 3)

        return: (B, 3)
        """

        quats = quats.reshape(-1, 4)
        vecs = vecs.reshape(-1, 3)
        qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        vx, vy, vz = vecs[:, 0], vecs[:, 1], vecs[:, 2]
        vqx = 2 * (qy * vz - qz * vy)
        vqy = 2 * (qz * vx - qx * vz)
        vqz = 2 * (qx * vy - qy * vx)
        vqw = qw * qw - qx * qx - qy * qy - qz * qz
        vqx += vqw * qx
        vqy += vqw * qy
        vqz += vqw * qz
        vqx += vqx
        vqy += vqy
        vqz += vqz
        vx += vqy * qz - vqz * qy
        vy += vqz * qx - vqx * qz
        vz += vqx * qy - vqy * qx
        vecs = torch.stack([vx, vy, vz], dim=1)

        return vecs
    
    def invert(self):
        """
        Inverses the entire Rotation object.

        return: Rotation
        """

        return Rotation(
            rots=Rotation.invert_rot(self.rots),
            quats=Rotation.invert_quat(self.quats)
        )

    def __getitem__(self, index):
        if type(index) != Tuple:
            index = (index,)

        return Rotation(
            rots=self.rots[index],
            quats=self.quats[index]
        )

    def __mul__(self, right: torch.Tensor):
        rots = self.rots * right[..., None, None]
        quats = self.quats * right[..., None]

        return Rotation(rots=rots, quats=quats)

    def __rmul__(self, left: torch.Tensor):
        return self.__mul__(left)
    
    def get_rots(self):
        return self.rots

    def get_quats(self):
        return self.quats

    def apply(self, vecs: torch.Tensor):
        """
        Applies the rotation to a batch of vectors.

        vecs: (B, 3)

        return: (B, 3)
        """

        return Rotation.mul_quats_vecs(self.quats, vecs)

    def invert_apply(self, vecs: torch.Tensor):
        """
        Applies the inverse rotation to a batch of vectors.

        vecs: (B, 3)

        return: (B, 3)
        """

        return Rotation.mul_quats_vecs(self.invert_quat(self.quats), vecs)
    
    def compose_q_update_vec(self, q_update_vec: torch.Tensor):
        """
        q_update_vec: (B, 3) -> quaternion update tensor

        return: Rotation
        """
        quats = self.get_quats()
        new_quats = quats + Rotation.mul_quats_vecs(quats, q_update_vec)

        return Rotation(quats=new_quats)

    def compose_r(self, r):
        """
        Compose the rotation matrices of the current Rotation object with another.

        r: Rotation

        return: Rotation
        """

        r1 = self.get_rots()
        r2 = r.get_rots()
        new_rot_mats = r1 @ r2

        return Rotation(rots=new_rot_mats)

    def compose_q(self, r):
        """
        Compose the quaternions of the current Rotation object with another.

        r: Rotation

        return: Rotation
        """

        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = Rotation.mul_quats(q1, q2)

        return Rotation(quats=new_quats)
    
    def cuda(self):
        return Rotation(self.rots.cuda(), self.quats.cuda())


class Rigid:
    def __init__(self, rots: Rotation, trans: torch.Tensor):
        """
        rots: Rotation
        trans: (B, 3), torch.Tensor
        """

        self.rots = rots
        self.trans = trans

    def __getitem__(self, index):
        return Rigid(
            self.rots[index],
            self.trans[index + (slice(None),)]
        )

    def __mul__(self, right: torch.Tensor):
        new_rots = self.rots * right
        new_trans = self.trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor):
        return self.__mul__(left)

    def get_rots(self):
        return self.rots

    def get_trans(self):
        return self.trans

    def compose_q_update_vec(self, q_update_vec):
        """
        Composes the transformation with a quaternion update vector.
        """

        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self.rots.compose_q_update_vec(q_vec)

        new_trans = self.rots.apply(t_vec)
        new_translation = self.trans + new_trans

        return Rigid(new_rots, new_translation)

    def compose(self, r):
        """
        Composes the current rigid object with another.

        r: Rigid

        return: Rigid (the composition of the two transformations)
        """

        new_rot = self.rots.compose_r(r.rots)
        new_trans = self.rots.apply(r.trans) + self.trans

        return Rigid(new_rot, new_trans)

    def apply(self, vecs):
        """
        Applies the transformation to a coordinate tensor.

        vecs: (B, 3)

        return: (B, 3)
        """

        return self.rots.apply(vecs) + self.trans

    def invert_apply(self, vecs):
        """
        Applies the inverse transformation to a coordinate tensor.

        vecs: (B, 3)

        return: (B, 3)
        """

        return self.rots.invert_apply(vecs - self.trans)

    def invert(self):
        rot_inv = self.rots.invert()
        trans_inv = rot_inv.apply(self.trans) * -1

        return Rigid(rot_inv, trans_inv)

    def from3points(x1, x2, x3):
        """
        Algorithm 21: Rigid from 3 points using the Gram-Schmidt process

        x1, x2, x3: (B, 3) coordinate tensors

        return: Rigid (transformation object)
        """

        v1 = x3 - x2
        v2 = x1 - x2
        e1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
        u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
        e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
        e3 = torch.cross(e1, e2, dim=-1)

        rots = torch.stack([e1, e2, e3], dim=1)
        trans = x2.clone()

        return Rigid(rots, trans)

    def __repr__(self):
        return f"<Rigid object with shape {self.rots.shape}>"

    def cuda(self):
        return Rigid(self.rots.cuda(), self.trans.cuda())