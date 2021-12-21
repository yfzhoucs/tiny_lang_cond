import numpy as np
from envs import reach_random_shape as reach
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


# # https://www.titanwolf.org/Network/q/8f901b91-0923-43ed-90fd-bf5f0143d9a1/y
# def disable_view_window():
#     from gym.envs.classic_control import rendering
#     org_constructor = rendering.Viewer.__init__

#     def constructor(self, *args, **kwargs):
#         org_constructor(self, *args, **kwargs)
#         self.window.set_visible(visible=False)

#     rendering.Viewer.__init__ = constructor


def l2(pos1, pos2):
    d = 0.
    for i in range(len(pos1)):
        d += (pos1[i] - pos2[i]) * (pos1[i] - pos2[i])
    return d


def differentiable_arm_model(list_link_lengths, list_joint_angles):
    assert len(list_link_lengths) == len(list_joint_angles)

    # init
    end_x = torch.zeros(1, requires_grad=True)
    end_y = torch.zeros(1, requires_grad=True)
    joint_positions_x = []
    joint_positions_y = []
    joint_positions_x.append(end_x)
    joint_positions_y.append(end_y)
    angles = []
    angles.append(0)

    # calculate the position of each joint
    for i in range(len(list_link_lengths)):
        angles.append(angles[-1] + list_joint_angles[i])
        joint_positions_x.append(list_link_lengths[i] * torch.cos(angles[-1]) + joint_positions_x[-1])
        joint_positions_y.append(list_link_lengths[i] * torch.sin(angles[-1]) + joint_positions_y[-1])

    return joint_positions_x[-1], joint_positions_y[-1]


def compute_target_angle(list_link_lengths, list_joint_angles, target_position):
    target_x = torch.tensor([target_position[0]], requires_grad=True)
    target_y = torch.tensor([target_position[1]], requires_grad=True)

    current_joint_angles = [torch.tensor(list_joint_angles[i], requires_grad=True) for i in
                            range(len(list_joint_angles))]
    optimizer_joint_angles = torch.optim.Adam(params=current_joint_angles)
    criterion = nn.L1Loss()

    for i in range(200):
        optimizer_joint_angles.zero_grad()
        end_x_pred, end_y_pred = differentiable_arm_model(list_link_lengths, current_joint_angles)
        loss = criterion(end_x_pred, target_x) + criterion(end_y_pred, target_y)
        loss.backward()
        optimizer_joint_angles.step()

    return torch.stack(current_joint_angles)


class Recorder:
    def __init__(self, data_dir, data_id):
        self.data_dir = data_dir
        self.data_id = data_id
        self.joints_seq = []
        self.end_positions_seq = []
        self.id_dir = os.path.join(data_dir, str(data_id))
        if not os.path.isdir(self.id_dir):
            os.mkdir(self.id_dir)

    def save_joints(self):
        joints_seq = np.stack(self.joints_seq)
        with open(self.id_dir + '/joints.npy', 'wb') as f:
            np.save(f, self.joints_seq)

    def save_end_positions(self):
        end_positions_seq = np.stack(self.end_positions_seq)
        with open(self.id_dir + '/end_positions.npy', 'wb') as f:
            np.save(f, self.end_positions_seq)

    def save_img(self, img, step):
        plt.imsave(self.id_dir + '/' + str(step) + '.png', img)

    def record_step(self, img, joints, end_position, step):
        self.save_img(img, step)
        self.joints_seq.append(np.array(joints, copy=True))
        self.end_positions_seq.append(np.array(end_position, copy=True))

    def record_goal(self, object_id, store_data_id=False):
        with open(self.id_dir + '/instructions.txt', 'a') as f:
            if store_data_id:
                f.write(f'{self.data_id} {object_id}\n')
            else:
                f.write(f'{object_id}\n')

    def record_object_positions(self, object_list, store_data_id=False):
        with open(self.id_dir + '/object_lists.txt', 'a') as f:
            if store_data_id:
                f.write(f'{self.data_id}, ')
            for i in range(len(object_list)):
                position_x = object_list[i][0]
                position_y = object_list[i][1]
                radius = object_list[i][2]
                f.write(f'{position_x}, {position_y}, {radius}, ')
            f.write(f'\n')

    def record_user_input(self, positive_action, negative_action):
        with open(os.path.join(self.id_dir, 'positive_action_user_inputs.txt'), 'a') as f:
            f.write(f'{self.data_id}, {positive_action}\n')
        with open(os.path.join(self.id_dir, 'negative_action_user_inputs.txt'), 'a') as f:
            f.write(f'{self.data_id}, {negative_action}\n')


# https://stackoverflow.com/questions/46996866/sampling-uniformly-within-the-unit-circle
def even_distribution_in_a_circle(circle_radius=50):
    length = np.sqrt(np.random.uniform(0, 1))
    angle = np.pi * np.random.uniform(0, 2)

    x = circle_radius * np.cos(angle) * length
    y = circle_radius * np.sin(angle) * length

    return x, y


def collect_sequence_data(data_id, screen_width, screen_height, data_dir, disable_window=True, step_threshold=250):
    # Create an environment
    robot = reach.SimpleRobot([30., 30.])
    object_geom_list = []
    num_objs = np.random.randint(3, 5)
    for obj_id in range(num_objs):
        # [x, y, radius, (r, g, b), shape]
        my_obj = [*even_distribution_in_a_circle(circle_radius=50), np.random.uniform(5, 10),
                  np.eye(3)[np.random.randint(3)], np.random.randint(3)],
        object_geom_list.append(*my_obj)
    object_geom_list = reach.ObjectList(
        object_geom_list, screen_width=screen_width, screen_height=screen_height)
    env = reach.Reach(
        robot=robot, object_list=object_geom_list,
        screen_width=screen_width, screen_height=screen_height)

    # if disable_window:
    #     disable_view_window()
    object_list = object_geom_list.get_objects()

    # Select a goal
    num_objs = len(object_list)
    object_id = np.random.randint(num_objs)
    target_x = object_list[object_id][0]
    target_y = object_list[object_id][1]
    error_limit = object_list[object_id][2] / 2

    # Start acting and recording
    recorder = Recorder(data_dir, data_id)
    recorder.record_goal(object_id)
    recorder.record_object_positions(object_list)
    joints = env.reset(np.random.random((len(robot.lengths),)) * np.pi * 2)
    img = env.render(mode="rgb_array")
    step = 0

    # Record initial joints and image
    recorder.record_step(img, joints, robot.get_end_position(), step)

    while l2(robot.get_end_position(), (target_x, target_y)) > error_limit * error_limit:
        # Moving the end towards the goal
        # Calculate target action
        action = compute_target_angle(robot.lengths, joints, (target_x, target_y))

        # Execute action
        joints, _, _, _ = env.step(action)
        img = env.render(mode="rgb_array")

        # Count steps
        step += 1

        # Record joints and image
        recorder.record_step(img, joints, robot.get_end_position(), step)

        if step >= step_threshold:
            break

    recorder.save_joints()
    recorder.save_end_positions()
    positive_action_input = input("\nPlease describe action robot did as POSITIVE action\n")
    negative_action_input = input("\nPlease describe action robot did as NEGATIVE action\n")
    recorder.record_user_input(positive_action_input, negative_action_input)
    print(data_id, step)

    env.close()


if __name__ == '__main__':
    screen_width = 128
    screen_height = 128
    data_dir = './data_position_random_shape_30_20_part1/'

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    for i in range(0, 1000):
        data_id = i
        collect_sequence_data(data_id, screen_width, screen_height, data_dir)
