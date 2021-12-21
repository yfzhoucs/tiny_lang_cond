import os
import pyglet


def display_using_pyglet(image_files):
    pyg_arr = []
    for image_file in image_files:
        pyg_arr.append(pyglet.resource.image(image_file))
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


def get_action_user_input(folder_path: str):
    experiments = [os.path.join(folder_path, o) for o in os.listdir(folder_path) if
                   os.path.isdir(os.path.join(folder_path, o))]
    for folder in experiments:
        fp_in = [os.path.join(folder, o) for o in os.listdir(folder) if o.lower().endswith(".png")]
        image_files = sorted(fp_in, key=lambda x: int(x.split(os.path.sep)[-1].lower().split(".png")[0]))
        display_using_pyglet(image_files)


if __name__ == '__main__':
    experiment_folder = "data_position_random_shape_30_20_part1"
    get_action_user_input(experiment_folder)
