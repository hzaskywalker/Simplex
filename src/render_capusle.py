import numpy as np
from robot.renderer import Renderer

a = Renderer()
a.capsule(1, 0.2, (255, 255, 255), np.eye(4))
a.axis(np.eye(4), 1)

a.set_camera_position(-10, 0, 0)
a.set_camera_rotation(0, 0)
a.render(mode='interactive')