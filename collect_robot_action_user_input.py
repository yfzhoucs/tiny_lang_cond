import json
import os
import pyglet


def display_using_pyglet(image_files):
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
        window.clear()
        sprite.draw()

    @window.event
    def on_key_press(symbol, modifier):
        # close the window
        window.close()

    pyglet.app.run()
    pyglet.app.exit()


def record_user_input(id_dir, username, positive_action, negative_action):
    experiment_user_input = {
        username: [positive_action, negative_action]
    }
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
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(abs_dir, folder_path)
    experiments = [os.path.join(folder_path, o) for o in os.listdir(folder_path) if
                   os.path.isdir(os.path.join(folder_path, o))]
    for folder in experiments:
        fp_in = [os.path.join(folder, o) for o in os.listdir(folder) if o.lower().endswith(".png")]
        image_files = sorted(fp_in, key=lambda x: int(x.split(os.path.sep)[-1].lower().split(".png")[0]))
        if not check_json_for_username(folder, username):
            continue
        display_using_pyglet(image_files)
        positive_action_input = input("\nPlease describe action robot did as POSITIVE action\n")
        negative_action_input = input("\nPlease describe action robot did as NEGATIVE action\n")
        record_user_input(folder, username, positive_action_input, negative_action_input)


if __name__ == '__main__':
    experiment_folder = "data_position_random_shape_30_20_part1"
    user_name = input("\nPlease provide your username (It will be used to store data in JSON):\n")
    get_action_user_input(experiment_folder, user_name)