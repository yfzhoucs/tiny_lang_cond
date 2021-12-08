from collect_data import l2
from envs import reach
from models.backbone_across_robot_inject_trial import Backbone
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib.animation import PillowWriter


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')


fig = plt.figure()
task_dict = {
    0: 'Red',
    1: 'Green',
    2: 'Blue'
}


# https://stackoverflow.com/questions/46996866/sampling-uniformly-within-the-unit-circle
def even_distribution_in_a_circle(circle_radius=50):
    length = np.sqrt(np.random.uniform(0, 1))
    angle = np.pi * np.random.uniform(0, 2)

    x = circle_radius * np.cos(angle)
    y = circle_radius * np.sin(angle)

    return x, y


def convert_obversations_to_torch(joints, img, accumulate_angles=False):
    joints = np.reshape(joints, (1, 2))
    if accumulate_angles:
        joints[0][1] = joints[0][1] + joints[0][0]
    transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    joints = np.dot(joints, transform_matrix)
    for i in range(2):
        joints[:, 2 * i] = np.sin(joints[:, 2 * i])
        joints[:, 2 * i + 1] = np.cos(joints[:, 2 * i + 1])
    joints = torch.tensor(joints, dtype=torch.float32)

    img = torch.tensor(np.array([img]) / 255, dtype=torch.float32)
    return joints, img


def convert_action_to_numpy(action, current_joint_angles=None, use_delta=False, accumulate_angles=False):
    action = action.detach().numpy()

    if use_delta:
        current_joint_angles = current_joint_angles.to('cpu').numpy()
        action = action / 100
        action = current_joint_angles + action

    # Convert from sin-cos space to rad space
    action_rad = np.array([0., 0.])
    for i in range(2):
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


def execution_loop(model, env, robot, task_id, target_x, target_y, use_delta, error_limit=5, accumulate_angles=False, count=0):


    task_id = torch.tensor([task_id], dtype=torch.int32)
    model = model.eval()

    # Get observations from t_0
    joints = env.reset(np.random.random((2,)) * np.pi * 2)
    img = env.render(mode="rgb_array")
    squared_error = l2(robot.get_end_position(), (target_x, target_y))

    # Start looping
    step = 0
    ims = []
    ax = plt.imshow(img)
    # txt = plt.text(0, 0, str(step))
    while step < 300:

        # Predict action from the model
        joints, img = convert_obversations_to_torch(joints, img, accumulate_angles)
        img = img.to(device)
        joints = joints.to(device)
        task_id = task_id.to(device)        

        # 0, 1, 2 == r, g, b
        if step < 25:
            target_pos_embed = torch.tensor(np.load('0.npy')).to(device)
            mytxt = plt.text(0, -10, f'inject: target == red', fontsize=20)
            action, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred, target_pos_embed = model(img, joints, task_id, target_pos_embed=target_pos_embed)
        elif step < 50:
            target_pos_embed = torch.tensor(np.load('1.npy')).to(device)
            mytxt = plt.text(0, -10, f'inject: target == green', fontsize=20)
            action, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred, target_pos_embed = model(img, joints, task_id, target_pos_embed=target_pos_embed)
        elif step < 75:
            target_pos_embed = torch.tensor(np.load('2.npy')).to(device)
            mytxt = plt.text(0, -10, f'inject: target == blue', fontsize=20)
            action, target_position_pred, displacement_pred, attn_map, attn_map2, attn_map3, attn_map4, joints_pred, target_pos_embed = model(img, joints, task_id, target_pos_embed=target_pos_embed)
        else:
            break

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
        
        ax = plt.imshow(img)
        # txt.set_text(str(step))
        ims.append([ax, mytxt])

    decision = int(input('is this good?'))
    print(decision, decision==1)
    if decision == 1:
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
        ani.save(f"mind_injection_{count}.gif", writer='imagemagick')
    return 0


def main(model_path, use_delta, input_goal=False, accumulate_angles=False, count=0):
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
    if not input_goal:
        goal = np.random.randint(3)
    else:
        goal = int(input('please input goal (rgb=012): '))
    target_x = object_geom_list.get_objects()[goal][0]
    target_y = object_geom_list.get_objects()[goal][1]

    # load model
    model = Backbone(128, 2, 3, 192, add_displacement=True, device=device).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)

    # Execution loop
    success = execution_loop(model, env, robot, goal, target_x, target_y, use_delta, accumulate_angles=accumulate_angles, count=count)

    # Close the environment
    env.close()

    return success


if __name__ == '__main__':
    model_path = '/share/yzhou298/ckpts/train17-1-across-robot-trial/46.pth'
    use_delta = True
    accumulate_angles = False
    trials = 5
    success = 0
    for i in range(0, 10):
        success += main(model_path, use_delta, input_goal=False, accumulate_angles=accumulate_angles, count=i)
        print(i + 1, success)
    success_rate = success / trials
    print(success_rate)