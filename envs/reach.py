import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
# some functions are from https://github.com/AtsushiSakai/PythonRobotics
# some functions are from openai gym
import os
import sys

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

# from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL installed. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )



class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), width=1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(width)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class SimpleRobot():
    def __init__(self, lengths=np.array([100., 100.]), joints=None):
        self.lengths = np.array(lengths)
        if joints is None:
            self.joints = np.zeros((self.lengths.shape[0],))
        else:
            self.joints = np.array(joints)

        assert self.lengths.shape[0] == self.joints.shape[0]

    def get_end_position(self):
        start_x = 0
        start_y = 0
        end_x = 0
        end_y = 0
        angle = 0
        for i in range(self.joints.shape[0]):
            start_x = end_x
            start_y = end_y
            angle = angle + self.joints[i]
            end_x = start_x + np.cos(angle) * self.lengths[i]
            end_y = start_y + np.sin(angle) * self.lengths[i]
        return end_x, end_y


class ObjectList():
    def __init__(self, list_of_objects=[[50, 50, 5, (1, 0, 0)]], screen_width=256, screen_height=256):
        #[[x1, y1, r1, (r1, g1, b1)], [x2, y2, r2, (r2, g2, b2)], ...]
        self.list_of_object_geoms = []
        for i in range(len(list_of_objects)):
            circle = rendering.make_circle(list_of_objects[i][2])
            circletrans = rendering.Transform(translation=(list_of_objects[i][0] + screen_width / 2, list_of_objects[i][1] + screen_height / 2))
            circle.add_attr(circletrans)
            circle.set_color(*list_of_objects[i][3])
            self.list_of_object_geoms.append(circle)
        self.list_of_objects = list_of_objects

    def get_object_geoms(self):
        return self.list_of_object_geoms

    def get_objects(self):
        return self.list_of_objects

    def __len__(self):
        return len(self.list_of_objects)


class Reach(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, robot=SimpleRobot(), object_list=ObjectList(), screen_width=256, screen_height=256):
        # The robot
        self.robot = robot
        self.observation_space = spaces.Box(low=0, high=np.pi*2, shape=robot.joints.shape)
        self.action_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=robot.joints.shape)

        # Objects
        self.objects = object_list.get_object_geoms()

        # Similation parameters
        self.Kp = 15
        self.dt = 0.01
        self.viewer = None
        self.screen_width = screen_width
        self.screen_height = screen_height

    def step(self, action):
        # action is target angles of the joints
        assert action.shape[0] == self.robot.joints.shape[0]
        for i in range(action.shape[0]):
            self.robot.joints[i] = self.robot.joints[i] + self.Kp * self._ang_diff(action[i], self.robot.joints[i]) * self.dt
        return self.robot.joints, 0, False, {}

    def reset(self, joints=np.zeros((2,))):
        self.robot.joints = joints
        return self.robot.joints

    def render(self, mode='human'):
        # code from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

        # if not initialized, initialize it
        if self.viewer is None:

            # set up a window
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        for i in range(len(self.objects)):
            self.viewer.add_onetime(self.objects[i])

        # shoulder, arm, etc.
        start_x = 0
        start_y = 0
        end_x = self.screen_width / 2
        end_y = self.screen_height / 2
        angle = 0
        for i in range(self.robot.joints.shape[0]):
            start_x = end_x
            start_y = end_y
            angle = angle + self.robot.joints[i]
            end_x = start_x + np.cos(angle) * self.robot.lengths[i]
            end_y = start_y + np.sin(angle) * self.robot.lengths[i]
            line = Line(start=(start_x, start_y), end=(end_x, end_y), width=2)
            self.viewer.add_onetime(line)
            joint = rendering.make_circle(3)
            jointtrans = rendering.Transform(translation=(start_x, start_y))
            joint.add_attr(jointtrans)
            self.viewer.add_onetime(joint)


        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _ang_diff(self, theta1, theta2):
        # Returns the difference between two angles in the range -pi to +pi
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    objects = [
        [50, 50, 5, (1, 0, 0)],
        [-50, 80, 5, (0, 1, 0)],
        [30, -50, 5, (0, 0, 1)]
    ]
    env = Reach(SimpleRobot([50., 40.]), ObjectList(objects))
    state = env.reset()
    while True:
        for i in range(20):
            env.render()
            state, _, _, _ = env.step(np.array([5, 0]))
        for i in range(20):
            env.render()
            state, _, _, _ = env.step(np.array([5, 3]))
        for i in range(20):
            env.render()
            state, _, _, _ = env.step(np.array([5, 6]))

    env.close()

