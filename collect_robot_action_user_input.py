import json
import os
import pyglet


def display_using_pyglet(image_files):
    """
    Display images as GIF kind of format
    :param image_files: Image paths
    :return:
    """
    pyg_arr = []
    base_dir = os.path.dirname(image_files[0])
    pyglet.resource.path = [base_dir]
    pyglet.resource.reindex()
    for image_file in image_files:
        pyg_arr.append(pyglet.resource.image(os.path.basename(image_file)))
    animation = pyglet.image.Animation.from_image_sequence(pyg_arr, duration=0.1, loop=False)
    sprite = pyglet.sprite.Sprite(img=animation)
    w = sprite.width
    h = sprite.height
    window = pyglet.window.Window(width=w, height=h)
    r, g, b, alpha = 0.5, 0.5, 0.8, 0.5
    pyglet.gl.glClearColor(r, g, b, alpha)

    @window.event
    def on_draw():
        # draw sprite on window
        window.clear()
        sprite.draw()

    @window.event
    def on_key_press(symbol, modifier):
        # close the window
        window.close()
        sprite.delete()

    def close(event):
        # callback for closing window after some secocnds
        window.close()

    # pyglet.clock.schedule_once(close, 9.0)
    pyglet.app.run()
    # mendatory deletion of object to avoid repetition of previous images
    del animation
    del sprite


def record_user_input(id_dir, username, positive_action, negative_action):
    # user input dictionary
    experiment_user_input = {
        username: [positive_action, negative_action]
    }

    # Create a new user entry and add user input
    user_input_file = os.path.join(id_dir, 'user_inputs.json')
    if os.path.isfile(user_input_file):
        with open(user_input_file, 'r') as f:
            data = json.load(f)
            data[username] = experiment_user_input[username]
        with open(user_input_file, 'w') as f:
            json.dump(data, f)
    else:
        with open(user_input_file, 'w') as f:
            json.dump(experiment_user_input, f)


def check_json_for_username(id_dir: os.path, username: str):
    # check the entry for user in the json file
    user_input_file = os.path.join(id_dir, 'user_inputs.json')
    print(user_input_file)
    if os.path.isfile(user_input_file):
        with open(user_input_file, 'r') as f:
            data = json.load(f)
            if username in data.keys():
                print("We already have input for you!\n Here is your previous input:", data[username])
                return False
    return True


def get_action_user_input(folder_path: str, username: str):
    # make the experiment path absolute with respect to OS
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(abs_dir, folder_path)
    experiments = [os.path.join(folder_path, o) for o in os.listdir(folder_path) if
                   os.path.isdir(os.path.join(folder_path, o))]

    # Go through each trial and record the user input
    for folder in experiments:
        fp_in = [os.path.join(folder, o) for o in os.listdir(folder) if o.lower().endswith(".png")]
        image_files = sorted(fp_in, key=lambda x: int(x.split(os.path.sep)[-1].lower().split(".png")[0]))
        if not check_json_for_username(folder, username):
            continue

        # displays the images to user
        display_using_pyglet(image_files)

        # take user input
        positive_action_input = input("\nPlease describe action robot did as POSITIVE action\n")
        negative_action_input = input("\nPlease describe action robot did as NEGATIVE action\n")
        record_user_input(folder, username, positive_action_input, negative_action_input)


if __name__ == '__main__':
    # Ask for the experiments folder path
    experiment_folder = "data_position_random_shape_30_20_part1"
    user_name = input("\nPlease provide your username (It will be used to store data in JSON):\n")
    get_action_user_input(experiment_folder, user_name)
