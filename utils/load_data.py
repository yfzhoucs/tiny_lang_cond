import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt


class RobotDataset(Dataset):
    def __init__(self, data_dir, task_txt='instructions.txt', joints_log='joints.npy', 
        use_trigonometric_representation=False, use_delta=False, normalize=False,
        ending_angles_as_next=False):

        # Find all the task instructions
        instructions_by_trace_id = {}
        task_inst_fd = open(os.path.join(data_dir, task_txt), 'r')
        for line in task_inst_fd:
            tokens = line.split(' ')
            instructions_by_trace_id[int(tokens[0])] = int(tokens[1])

        # Find all the images and joints
        self.img_paths = []
        self.joints = []
        self.next_joints = []
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        all_dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        count = 0
        self.instructions = []
        for trace in all_dirs:
            count += 1
            joints_log_np = np.load(os.path.join(trace, joints_log))
            joints_log_np[joints_log_np < 0] += 2 * np.pi
            joints_log_np[joints_log_np > 2 * np.pi] -= 2 * np.pi
            self.joints.append(joints_log_np[:-1])
            if not ending_angles_as_next:
                self.next_joints.append(joints_log_np[1:])
            else:
                self.next_joints.append(np.tile(joints_log_np[-1:], (joints_log_np.shape[0] - 1, 1)))
            for i in range(joints_log_np.shape[0] - 1):
                self.img_paths.append(os.path.join(trace, str(i) + '.png'))
            trace_id = self._get_trace_id(trace)
            trace_length = joints_log_np.shape[0] - 1
            self.instructions.extend([instructions_by_trace_id[trace_id]] * trace_length)
            # if count == 1:
            #     print(self.img_paths)
            #     print(len(self.instructions))
            #     break

        self.joints = np.concatenate(self.joints)

        # If use trigonometric, which means to use sin(theta) and cos(theta) to discribe theta. 
        # This adds smoothness to the outputs.
        if use_trigonometric_representation:
            transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
            self.joints = np.dot(self.joints, transform_matrix)
            for i in range(2):
                self.joints[:, 2 * i] = np.sin(self.joints[:, 2 * i])
                self.joints[:, 2 * i + 1] = np.cos(self.joints[:, 2 * i + 1])
        self.next_joints = np.concatenate(self.next_joints)
        if use_trigonometric_representation:
            transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
            self.next_joints = np.dot(self.next_joints, transform_matrix)
            for i in range(2):
                self.next_joints[:, 2 * i] = np.sin(self.next_joints[:, 2 * i])
                self.next_joints[:, 2 * i + 1] = np.cos(self.next_joints[:, 2 * i + 1])

        # If use delta, which means that the output is the distance to move, instead of the position
        if use_delta:
            self.next_joints = self.next_joints - self.joints

        # If normalize data.
        if normalize:
            std_joints = np.std(self.joints)
            mean_joints = np.mean(self.joints)
            self.joints = (self.joints - mean_joints) / std_joints
            std_next_joints = np.std(self.next_joints)
            mean_next_joints = np.mean(self.next_joints)
            self.next_joints = (self.next_joints - mean_next_joints) / std_next_joints
            print('std:', std_next_joints, 'mean:', mean_next_joints)
            input()

        print(self.joints, self.next_joints)
        print(len(self.joints), len(self.img_paths), len(self.instructions), len(self.next_joints))
    
    def _get_trace_id(self, trace_dir):
        trace_id = int(trace_dir.strip(r'.png').split(r'/')[-1])
        return trace_id

    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, index):
        img = torch.tensor(io.imread(self.img_paths[index])[:,:,:3] / 255, dtype=torch.float32)
        joints = torch.tensor(self.joints[index], dtype=torch.float32)
        task_id = torch.tensor(self.instructions[index], dtype=torch.int32)
        next_joints = torch.tensor(self.next_joints[index], dtype=torch.float32)
        return img, joints, task_id, next_joints


class ImgToJointDataset(Dataset):
    def __init__(self, data_dir, joints_log='joints.npy', use_trigonometric_representation=False):

        # Find all the images and joints
        self.img_paths = []
        self.joints = []
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        all_dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        count = 0
        for trace in all_dirs:
            count += 1
            joints_log_np = np.load(os.path.join(trace, joints_log))
            joints_log_np[joints_log_np < 0] += 2 * np.pi
            joints_log_np[joints_log_np > 2 * np.pi] -= 2 * np.pi
            self.joints.append(joints_log_np[:])
            for i in range(joints_log_np.shape[0]):
                self.img_paths.append(os.path.join(trace, str(i) + '.png'))
            # if count == 2:
            #     print(self.img_paths)
            #     print(self.joints)
            #     break

        self.joints = np.concatenate(self.joints)
        if use_trigonometric_representation:
            transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
            self.joints = np.dot(self.joints, transform_matrix)
            # print(self.joints)
            for i in range(2):
                self.joints[:, 2 * i] = np.sin(self.joints[:, 2 * i])
                self.joints[:, 2 * i + 1] = np.cos(self.joints[:, 2 * i + 1])
            # print(self.joints)
        print(len(self.joints), len(self.img_paths))

    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, index):
        img = torch.tensor(io.imread(self.img_paths[index])[:,:,:3] / 255, dtype=torch.float32)
        joints = torch.tensor(self.joints[index], dtype=torch.float32)
        return img, joints


class ComprehensiveVisualDataset(Dataset):
    def __init__(self, data_dir, joints_log='joints.npy', 
        end_position_log='end_positions.npy', 
        object_list_txt='object_lists.txt',
        use_trigonometric_representation=False):

        # Collect all object lists
        object_lists_by_trace_id = {}
        object_lists_fd = open(os.path.join(data_dir, object_list_txt), 'r')
        for line in object_lists_fd:
            tokens = line.strip().strip(',').split(', ')
            num_objects = len(tokens) // 3
            objects_in_this_trace = []
            for i in range(num_objects):
                objects_in_this_trace.append(float(tokens[i * 3 + 1]))
                objects_in_this_trace.append(float(tokens[i * 3 + 2]))
            trace_id = int(tokens[0])
            object_lists_by_trace_id[trace_id] = objects_in_this_trace
        # print(object_lists_by_trace_id)

        # Find all the images and joints
        self.img_paths = []
        self.joints = []
        self.end_positions = []
        self.object_lists = []
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        all_dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        count = 0
        for trace in all_dirs:
            count += 1

            # Collect joint angles
            joints_log_np = np.load(os.path.join(trace, joints_log))
            joints_log_np[joints_log_np < 0] += 2 * np.pi
            joints_log_np[joints_log_np > 2 * np.pi] -= 2 * np.pi
            self.joints.append(joints_log_np[:])

            # Collect end positions
            end_positions_log_np = np.load(os.path.join(trace, end_position_log))
            self.end_positions.append(end_positions_log_np[:])

            # Collect img paths
            for i in range(joints_log_np.shape[0]):
                self.img_paths.append(os.path.join(trace, str(i) + '.png'))
            
            # Add object lists
            trace_length = joints_log_np.shape[0]
            trace_id = self._get_trace_id(trace)
            self.object_lists.extend([object_lists_by_trace_id[trace_id]] * trace_length)

            # Debugging printing
            # if count == 2:
            #     print(self.img_paths)
            #     print(self.joints)
            #     break

        self.joints = np.concatenate(self.joints)
        self.end_positions = np.concatenate(self.end_positions)
        self.object_lists = np.array(self.object_lists)
        # print(self.object_lists[-50:])
        # print(self.object_lists.shape)
        # print(self.img_paths[-50:])

        # If use trigo mode
        if use_trigonometric_representation:
            transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
            self.joints = np.dot(self.joints, transform_matrix)
            # print(self.joints)
            for i in range(2):
                self.joints[:, 2 * i] = np.sin(self.joints[:, 2 * i])
                self.joints[:, 2 * i + 1] = np.cos(self.joints[:, 2 * i + 1])
            # print(self.joints)
        print(len(self.joints), len(self.img_paths), len(self.end_positions))
    
    def _get_trace_id(self, trace_dir):
        trace_id = int(trace_dir.strip(r'.png').split(r'/')[-1])
        return trace_id

    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, index):
        img = torch.tensor(io.imread(self.img_paths[index])[:,:,:3] / 255, dtype=torch.float32)
        joints = torch.tensor(self.joints[index], dtype=torch.float32)
        end_position = torch.tensor(self.end_positions[index], dtype=torch.float32)
        object_list = torch.tensor(self.object_lists[index], dtype=torch.float32)
        return img, joints, end_position, object_list


class TargetPositionlDataset(Dataset):
    def __init__(self, data_dir, joints_log='joints.npy', 
        end_position_log='end_positions.npy', 
        object_list_txt='object_lists.txt',
        task_txt='instructions.txt',
        use_trigonometric_representation=False):

        # Collect all object lists
        object_lists_by_trace_id = {}
        object_lists_fd = open(os.path.join(data_dir, object_list_txt), 'r')
        for line in object_lists_fd:
            tokens = line.strip().strip(',').split(', ')
            num_objects = len(tokens) // 3
            objects_in_this_trace = []
            for i in range(num_objects):
                objects_in_this_trace.append(float(tokens[i * 3 + 1]))
                objects_in_this_trace.append(float(tokens[i * 3 + 2]))
            trace_id = int(tokens[0])
            object_lists_by_trace_id[trace_id] = objects_in_this_trace

        # Collect all instructions
        instructions_by_trace_id = {}
        task_inst_fd = open(os.path.join(data_dir, task_txt), 'r')
        for line in task_inst_fd:
            tokens = line.split(' ')
            instructions_by_trace_id[int(tokens[0])] = int(tokens[1])

        # Find all the images and joints
        self.img_paths = []
        self.joints = []
        self.end_positions = []
        self.object_lists = []
        self.instructions = []
        self.target_positions = []
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        all_dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        count = 0
        for trace in all_dirs:
            count += 1

            # Collect joint angles
            joints_log_np = np.load(os.path.join(trace, joints_log))
            joints_log_np[joints_log_np < 0] += 2 * np.pi
            joints_log_np[joints_log_np > 2 * np.pi] -= 2 * np.pi
            self.joints.append(joints_log_np[:])

            # Collect end positions
            end_positions_log_np = np.load(os.path.join(trace, end_position_log))
            self.end_positions.append(end_positions_log_np[:])

            # Collect img paths
            for i in range(joints_log_np.shape[0]):
                self.img_paths.append(os.path.join(trace, str(i) + '.png'))
            
            # Add object lists
            trace_length = joints_log_np.shape[0]
            trace_id = self._get_trace_id(trace)
            task_id = instructions_by_trace_id[trace_id]
            self.object_lists.extend([object_lists_by_trace_id[trace_id]] * trace_length)
            self.instructions.extend([task_id] * trace_length)
            self.target_positions.extend([object_lists_by_trace_id[trace_id][task_id * 2:task_id * 2 + 2]] * trace_length)

            # # Debugging printing
            # if count == 2:
            #     print(self.img_paths)
            #     # print(self.joints)
            #     break

        self.joints = np.concatenate(self.joints)
        self.end_positions = np.concatenate(self.end_positions)
        self.object_lists = np.array(self.object_lists)
        self.target_positions = np.array(self.target_positions)
        # print(self.object_lists[-50:])
        # print(self.object_lists.shape)
        # print(self.img_paths[-50:])

        # If use trigo mode
        if use_trigonometric_representation:
            transform_matrix = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
            self.joints = np.dot(self.joints, transform_matrix)
            # print(self.joints)
            for i in range(2):
                self.joints[:, 2 * i] = np.sin(self.joints[:, 2 * i])
                self.joints[:, 2 * i + 1] = np.cos(self.joints[:, 2 * i + 1])
            # print(self.joints)
            self.joints = self.joints * 100
        print(len(self.joints), len(self.img_paths), len(self.end_positions))
    
    def _get_trace_id(self, trace_dir):
        trace_id = int(trace_dir.strip(r'.png').split(r'/')[-1])
        return trace_id

    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, index):
        img = torch.tensor(io.imread(self.img_paths[index])[:,:,:3] / 255, dtype=torch.float32)
        joints = torch.tensor(self.joints[index], dtype=torch.float32)
        task_id = torch.tensor(self.instructions[index], dtype=torch.int32)
        end_position = torch.tensor(self.end_positions[index], dtype=torch.float32)
        object_list = torch.tensor(self.object_lists[index], dtype=torch.float32)
        target_position = torch.tensor(self.target_positions[index], dtype=torch.float32)
        return img, joints, task_id, end_position, object_list, target_position


class ComprehensiveRobotDataset(Dataset):
    def __init__(self, data_dir, task_txt='instructions.txt', joints_log='joints.npy', 
        end_position_log='end_positions.npy', 
        object_list_txt='object_lists.txt',
        use_trigonometric_representation=False, 
        use_delta=False, normalize=False,
        ending_angles_as_next=False, amplify_trigonometric=False,
        add_displacement=False, accumulate_angles=False):

        self.add_displacement = add_displacement

        # Collect all object lists
        object_lists_by_trace_id = {}
        object_lists_fd = open(os.path.join(data_dir, object_list_txt), 'r')
        for line in object_lists_fd:
            tokens = line.strip().strip(',').split(', ')
            num_objects = len(tokens) // 3
            objects_in_this_trace = []
            for i in range(num_objects):
                objects_in_this_trace.append(float(tokens[i * 3 + 1]))
                objects_in_this_trace.append(float(tokens[i * 3 + 2]))
            trace_id = int(tokens[0])
            object_lists_by_trace_id[trace_id] = objects_in_this_trace

        # Find all the task instructions
        instructions_by_trace_id = {}
        task_inst_fd = open(os.path.join(data_dir, task_txt), 'r')
        for line in task_inst_fd:
            tokens = line.split(' ')
            instructions_by_trace_id[int(tokens[0])] = int(tokens[1])

        # Find all the images and joints
        self.img_paths = []
        self.joints = []
        self.next_joints = []
        self.instructions = []
        self.end_positions = []
        self.object_lists = []
        self.target_positions = []
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
        all_dirs = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        count = 0
        for trace in all_dirs:

            # Count number
            count += 1

            # Load joints and next joints
            joints_log_np = np.load(os.path.join(trace, joints_log))
            if accumulate_angles:
                # print(joints_log_np.shape)
                joints_log_np[:, 1] = joints_log_np[:, 0] + joints_log_np[:, 1]
                # print(joints_log_np.shape)
            joints_log_np[joints_log_np < 0] += 2 * np.pi
            joints_log_np[joints_log_np > 2 * np.pi] -= 2 * np.pi
            self.joints.append(joints_log_np[:-1])
            
            # If use the end frame of the trace as next
            if not ending_angles_as_next:
                self.next_joints.append(joints_log_np[1:])
            else:
                self.next_joints.append(np.tile(joints_log_np[-1:], (joints_log_np.shape[0] - 1, 1)))
            
            # Collect end positions
            end_positions_log_np = np.load(os.path.join(trace, end_position_log))
            self.end_positions.append(end_positions_log_np[:-1])

            # Collect img paths
            for i in range(joints_log_np.shape[0] - 1):
                self.img_paths.append(os.path.join(trace, str(i) + '.png'))

            # Add object lists and instructions
            trace_length = joints_log_np.shape[0] - 1
            trace_id = self._get_trace_id(trace)
            task_id = instructions_by_trace_id[trace_id]
            self.object_lists.extend([object_lists_by_trace_id[trace_id]] * trace_length)
            self.instructions.extend([task_id] * trace_length)
            self.target_positions.extend([object_lists_by_trace_id[trace_id][task_id * 2:task_id * 2 + 2]] * trace_length)
            
            # if count == 1:
            #     print(self.img_paths)
            #     print(len(self.instructions))
            #     break

        # Convert lists of data to numpy arrays
        self.end_positions = np.concatenate(self.end_positions)
        self.object_lists = np.array(self.object_lists)
        self.target_positions = np.array(self.target_positions)
        if add_displacement:
            self.displacement = self.target_positions - self.end_positions

        # If use trigonometric, which means to use sin(theta) and cos(theta) to discribe theta. 
        # This adds smoothness to the outputs.
        self.joints = np.concatenate(self.joints)
        if use_trigonometric_representation:
            self.joints = np.repeat(self.joints, 2, axis=1)
            for i in range(self.joints.shape[1] // 2):
                self.joints[:, 2 * i] = np.sin(self.joints[:, 2 * i])
                self.joints[:, 2 * i + 1] = np.cos(self.joints[:, 2 * i + 1])
        self.next_joints = np.concatenate(self.next_joints)
        if use_trigonometric_representation:
            self.next_joints = np.repeat(self.next_joints, 2, axis=1)
            for i in range(self.joints.shape[1] // 2):
                self.next_joints[:, 2 * i] = np.sin(self.next_joints[:, 2 * i])
                self.next_joints[:, 2 * i + 1] = np.cos(self.next_joints[:, 2 * i + 1])

        # If amplify the trigonometric angles. This balances the absolute values of the joint angle loss
        if amplify_trigonometric:
            self.joints = self.joints * 100
            self.next_joints = self.next_joints * 100

        # If use delta, which means that the output is the distance to move, instead of the position
        if use_delta:
            self.next_joints = self.next_joints - self.joints

        # If normalize data.
        if normalize:
            std_joints = np.std(self.joints)
            mean_joints = np.mean(self.joints)
            self.joints = (self.joints - mean_joints) / std_joints
            std_next_joints = np.std(self.next_joints)
            mean_next_joints = np.mean(self.next_joints)
            self.next_joints = (self.next_joints - mean_next_joints) / std_next_joints
            print('std:', std_next_joints, 'mean:', mean_next_joints)
            input()

        # print(self.joints, self.next_joints)
        # print(len(self.joints), len(self.img_paths), len(self.instructions), len(self.next_joints))
    
    def _get_trace_id(self, trace_dir):
        trace_id = int(trace_dir.strip(r'.png').split(r'/')[-1])
        return trace_id

    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, index):
        img = torch.tensor(io.imread(self.img_paths[index])[:,:,:3] / 255, dtype=torch.float32)
        joints = torch.tensor(self.joints[index], dtype=torch.float32)
        task_id = torch.tensor(self.instructions[index], dtype=torch.int64)
        end_position = torch.tensor(self.end_positions[index], dtype=torch.float32)
        object_list = torch.tensor(self.object_lists[index], dtype=torch.float32)
        target_position = torch.tensor(self.target_positions[index], dtype=torch.float32)
        next_joints = torch.tensor(self.next_joints[index], dtype=torch.float32)
        if self.add_displacement:
            displacement = torch.tensor(self.displacement[index], dtype=torch.float32)
            return img, joints, task_id, end_position, object_list, target_position, next_joints, displacement
        return img, joints, task_id, end_position, object_list, target_position, next_joints


if __name__ == '__main__':
    # dataset = RobotDataset(data_dir='../data/',
    #     use_trigonometric_representation=True, use_delta=True, normalize=False,
    #     ending_angles_as_next=True)

    dataset = ComprehensiveRobotDataset(
        '../data_position/', use_trigonometric_representation=True, amplify_trigonometric=True)

    for img, joints, task_id, end_position, object_list, target_position, next_joints in dataset:
        print(joints, task_id, end_position, object_list, target_position, next_joints)
        # print(img)
        plt.imshow(img)
        plt.show()
        # input()
