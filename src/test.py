import numpy as np
import tqdm
from simplex_c import Simplex

def test1():
    sim = Simplex()
    box = sim.box(np.array([1, 1, 1]))
    box2 = sim.box(np.array([1, 1, 1]))
    sphere = sim.sphere(3)
    sphere.set_pose(np.eye(4)[None,:]+ np.zeros((256,))[:, None, None])
    sim.add_shape(box).add_shape(box2).add_shape(sphere)
    sim.add_shape(box).add_shape(box2).add_shape(sphere)
    sim.add_shape(box).add_shape(box2).add_shape(sphere)
    sim.add_shape(box2)

    for i in tqdm.trange(10000):
        pose = np.eye(4)[None,:] + np.zeros((256,))[:, None, None]
        box.set_pose(pose)
        pose[:, :3 ,3] = np.array([0, 0, 1])
        pose[:, :3,:3] += np.random.random(size=(256, 3, 3))
        box2.set_pose(pose)
        sim.collide()
        #box.contype=0

def test_capsule():
    sim = Simplex()
    box = sim.box(np.array([1,1,1]))
    capsule1 = sim.capsule(0.1, 1)
    capsule2 = sim.capsule(0.1, 1)

    sim.clear_shapes()
    sim.add_shape(box).add_shape(capsule1).add_shape(capsule2)

    box.set_pose(np.eye(4)[None,:])

    pose1 = np.eye(4)
    pose1[2, 3] = 0.6-1e-5
    capsule1.set_pose(pose1[None,:])

    pose2 = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    pose2[2, 3] = 1.
    capsule2.set_pose(pose2[None,:])

    box.contype=3
    capsule1.contype=1
    capsule2.contype=2
    sim.collide()
    print(sim.normal_pos)
    print(sim.batch)
    print(sim.object_pair)

def test_grad():
    sim = Simplex()
    box = sim.box(np.array([1, 1, 1]))
    box.set_pose(np.eye(4)[None,:])

    """
    box2 = sim.box(np.array([1, 1, 1]))
    pose = np.eye(4)[None,:]
    pose[:, 2, 3] = 1
    box2.set_pose(pose)
    """

    sphere = sim.sphere(0.5)
    sphere.set_pose(np.array([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.9],
            [0, 0, 0, 1],
        ]
    ]))
    sim.add_shape(sphere).add_shape(box)
    sim.collide(computeJacobian=True)
    jac = sim.jacobian.reshape(2, 2, 6, 7)

    d = np.zeros((sim.batch.shape[0], 7))
    d[:, 1:4] = 1
    sim.backward(d)
    print('sphere time', sphere.grad)
    print('box time', box.grad)


if __name__ == '__main__':
    #test_capsule()
    #test1()
    test_grad()