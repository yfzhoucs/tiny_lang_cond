from collect_data import l2
from envs import reach
from models.backbone_joint_head import Backbone
import numpy as np
import torch


# https://stackoverflow.com/questions/46996866/sampling-uniformly-within-the-unit-circle
def even_distribution_in_a_circle(circle_radius=50):
    length = np.sqrt(np.random.uniform(0, 1))
    angle = np.pi * np.random.uniform(0, 2)

    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)

    return x, y


def convert_obversations_to_torch(joints, img):
    joints = np.reshape(joints, (1, 2))
    transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    joints = np.dot(joints, transform_matrix)
    for i in range(2):
        joints[:, 2 * i] = np.sin(joints[:, 2 * i])
        joints[:, 2 * i + 1] = np.cos(joints[:, 2 * i + 1])
    joints = torch.tensor(joints, dtype=torch.float32)

    img = torch.tensor(np.array([img]) / 255, dtype=torch.float32)
    return joints, img


def convert_action_to_numpy(action):
    action = action.detach().numpy()
    action_rad = np.array([0, 0])
    for i in range(2):
        if action[0][i * 2 + 1] < -1:
            action[0][i * 2 + 1] = -1
        if action[0][i * 2 + 1] > 1:
            action[0][i * 2 + 1] = 1
        action_rad[i] = np.arccos(action[0][i * 2 + 1])
        if action[0][i * 2] < 0:
            action_rad[i] += np.pi - action_rad[i]
    return action_rad


def execution_loop(model, env, robot, task_id, target_x, target_y, error_limit=5):
    task_id = torch.tensor([task_id], dtype=torch.int32)

    # Get observations from t_0
    joints = env.reset(np.random.random((2,)) * np.pi * 2)
    img = env.render(mode="rgb_array")
    squared_error = l2(robot.get_end_position(), (target_x, target_y))

    # Start looping
    while squared_error > error_limit * error_limit:

        # Predict action from the model
        joints, img = convert_obversations_to_torch(joints, img)
        action, _ = model(img, joints, task_id)

        # Execute action
        action = convert_action_to_numpy(action)
        joints, _, _, _ = env.step(action)
        img = env.render(mode="rgb_array")

        # Compute error
        squared_error = l2(robot.get_end_position(), (target_x, target_y))
        print('error', np.sqrt(squared_error))
        input()
    print('done')
    input()


def main(model_path):
    # Create an environment
    screen_width = 128
    screen_height = 128
    robot = reach.SimpleRobot([30., 30.])
    object_geom_list = reach.ObjectList([
        # [x, y, radius, (r, g, b)]
        [*even_distribution_in_a_circle(circle_radius=50), 5., (1, 0, 0)],
        [*even_distribution_in_a_circle(circle_radius=50), 5., (0, 1, 0)],
        [*even_distribution_in_a_circle(circle_radius=50), 5., (0, 0, 1)]
    ], screen_width=screen_width, screen_height=screen_height)
    env = reach.Reach(
        robot=robot, object_list=object_geom_list,
        screen_width=screen_width, screen_height=screen_height)

    # Create goal to reach
    goal = int(input('please input goal to reach'))
    target_x = object_geom_list.get_objects()[goal][0]
    target_y = object_geom_list.get_objects()[goal][1]

    # load model
    model = Backbone(128, 2, 3, 128, device=torch.device('cpu'))
    model.load_state_dict(torch.load(model_path))

    # Execution loop
    execution_loop(model, env, robot, goal, target_x, target_y)

    # Close the environment
    env.close()


if __name__ == '__main__':
    model_path = '/share/yzhou298/train3-2-conv-img-encoder-relu-5-traces-aux-loss-joint-angle/0.pth'
    main(model_path)