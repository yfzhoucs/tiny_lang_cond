from collect_data import l2
from envs import reach
from models.backbone_across_robot import Backbone
import numpy as np
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')


# https://stackoverflow.com/questions/46996866/sampling-uniformly-within-the-unit-circle
def even_distribution_in_a_circle(circle_radius=50):
    length = np.sqrt(np.random.uniform(0, 1))
    angle = np.pi * np.random.uniform(0, 2)

    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)

    return x, y


def convert_obversations_to_torch(joints, img, accumulate_angles=False):
    joints = np.reshape(joints, (1, -1))
    if accumulate_angles:
        joints[0][1] = joints[0][1] + joints[0][0]
    joints = np.repeat(joints, 2, axis=1)
    for i in range(joints.shape[1] // 2):
        joints[:, 2 * i] = np.sin(joints[:, 2 * i])
        joints[:, 2 * i + 1] = np.cos(joints[:, 2 * i + 1])
    joints = torch.tensor(joints, dtype=torch.float32)

    img = torch.tensor(np.array([img]) / 255, dtype=torch.float32)
    return joints, img


def convert_action_to_numpy(action, current_joint_angles=None, use_delta=False, accumulate_angles=False):
    action = action.detach().numpy()
    if use_delta:
        current_joint_angles = current_joint_angles.to('cpu').numpy()
        action = action / 200
        action = current_joint_angles + action

    # Convert from sin-cos space to rad space
    action_rad = np.array([0. for i in range(action.shape[-1] // 2)])
    for i in range(action.shape[-1] // 2):
        # if action[0][i * 2 + 1] < -1:
        #     action[0][i * 2 + 1] = -1
        # if action[0][i * 2 + 1] > 1:
        #     action[0][i * 2 + 1] = 1
        action_rad[i] = np.arctan(action[0][i * 2] / action[0][i * 2 + 1])
        if action[0][i * 2 + 1] < 0:
            action_rad[i] += np.pi
        if action_rad[i] < 0:
            action_rad[i] += np.pi * 2
    # print('rad', action, action_rad)

    if accumulate_angles:
        action_rad[1] = action_rad[1] - action_rad[0]

    return action_rad


def execution_loop(model, env, robot, task_id, target_x, target_y, use_delta, error_limit=5, accumulate_angles=False):
    task_id = torch.tensor([task_id], dtype=torch.int32)
    model = model.eval()

    # Get observations from t_0
    joints = env.reset(np.random.random((robot.lengths.shape[0],)) * np.pi * 2)
    img = env.render(mode="rgb_array")
    squared_error = l2(robot.get_end_position(), (target_x, target_y))

    # Start looping
    step = 0
    while step < 300:

        # Predict action from the model
        joints, img = convert_obversations_to_torch(joints, img, accumulate_angles)
        img = img.to(device)
        joints = joints.to(device)
        task_id = task_id.to(device)
        attn_mask = np.ones((260, 260))
        attn_mask = torch.tensor(attn_mask, dtype=torch.float32).to(device)
        # action, target_position_pred, displacement_pred, displacement_embed, attn_map, joints_pred = model(img, joints, task_id, attn_mask=attn_mask)
        action, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred = model(img, joints, task_id)
        action = action.to('cpu')

        # Execute action
        action = convert_action_to_numpy(action, joints, use_delta, accumulate_angles)
        joints, _, _, _ = env.step(action)
        img = env.render(mode="rgb_array")

        # Compute error
        squared_error = l2(robot.get_end_position(), (target_x, target_y))

        step += 1
        print(f'displacement_pred {displacement_pred.detach().cpu().numpy()[0]}')
        print('error', step, np.sqrt(squared_error))
        # input()

        if squared_error < error_limit * error_limit:
            print('done')
            return 1
    return 0


def main(model_path, use_delta, input_goal=False, accumulate_angles=False):
    # Create an environment
    screen_width = 128
    screen_height = 128
    robot = reach.SimpleRobot([6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])
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
    if not input_goal:
        goal = np.random.randint(3)
    else:
        goal = int(input('please input goal (rgb=012): '))
    target_x = object_geom_list.get_objects()[goal][0]
    target_y = object_geom_list.get_objects()[goal][1]

    # load model
    model = Backbone(128, 10, 3, 192, add_displacement=True, device=device, ignore_default_2_joint_head=True).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)

    # Execution loop
    success = execution_loop(model, env, robot, goal, target_x, target_y, use_delta, accumulate_angles=accumulate_angles)

    # Close the environment
    env.close()

    return success


if __name__ == '__main__':
    model_path = '/share/yzhou298/ckpts/train17-1-across-robot-trial-transfer-6-6-6-6-6-6-6-6-6-6/105.pth'
    use_delta = False
    accumulate_angles = False
    trials = 100
    success = 0
    for i in range(trials):
        success += main(model_path, use_delta, input_goal=False, accumulate_angles=accumulate_angles)
        print(i + 1, success)
    success_rate = success / trials
    print(success_rate)