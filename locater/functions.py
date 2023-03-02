import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys

import cv2
import imagehash
import numpy as np
from Levenshtein import distance
from PIL import Image
from PySimpleGUI import WIN_CLOSED
from screeninfo import get_monitors
import PySimpleGUI as sg


def convert_nan(val):
    return "" if isinstance(val, float) and np.isnan(val) else str(val)


def is_image_product(file_path):
    # Load the image
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel
    has_alpha = image.shape[2] == 4
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) if has_alpha else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Increase the contrast by 50%
    image = np.interp(image, (0, 255), (-0.5, 0.5))
    image = np.clip(image * 1.5 + 0.5, 0, 255).astype(np.uint8)
    # Check if the top-left, top-right, bottom-left, and bottom-right pixels are either white or transparent
    if has_alpha:
        top_left = image[0, 0]
        bottom_right = image[-1, -1]
        is_white_or_transparent = (np.all(top_left[:3] >= (230, 230, 230)) or top_left[3] == 0) and (
                    np.all(bottom_right[:3] >= (230, 230, 230)) or bottom_right[3] == 0)
    else:
        top_left, top_right = image[0, 0], image[0, -1]
        bottom_left, bottom_right = image[-1, 0], image[-1, -1]
        is_white_or_transparent = np.all(top_left >= (230, 230, 230)) and np.all(bottom_right >= (230, 230, 230))

    return np.all(is_white_or_transparent)


# def is_image_product(file_path):
#     # Load the image
#     image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
#     # Check if the image has an alpha channel
#     has_alpha = image.shape[2] == 4
#     # Convert the image from BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) if has_alpha else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Increase the contrast by 50%
#     image = np.interp(image, (0, 255), (-0.5, 0.5))
#     image = np.clip(image * 1.5 + 0.5, 0, 255).astype(np.uint8)
#     # Get the top left and bottom right pixels of the image
#     top_left = image[0, 0]
#     bottom_right = image[-1, -1]
#     # Check if the top left and bottom right pixels are either white, transparent, or a combination of both
#     if has_alpha:
#         is_top_left_white_or_transparent = np.all(top_left[:3] >= (230, 230, 230)) or top_left[3] == 0
#         is_bottom_right_white_or_transparent = np.all(bottom_right[:3] >= (230, 230, 230)) or bottom_right[3] == 0
#     else:
#         is_top_left_white_or_transparent = np.all(top_left >= (230, 230, 230))
#         is_bottom_right_white_or_transparent = np.all(bottom_right >= (230, 230, 230))
#
#     return is_top_left_white_or_transparent and is_bottom_right_white_or_transparent


def add_top_level_parent_directory(file_path, folder_ref, folder_suffix="_background"):
    """
    adds a new top level directory to the path, just before the filename

    :param folder_suffix: after the folder name
    :param file_path: string file path
    :param folder_ref: new folder name
    :return: reconstructed file path
    """
    directory, file_name = os.path.split(file_path)
    new_directory = make_filepath_valid(os.path.join(directory, folder_ref + folder_suffix))
    new_file_path = os.path.join(new_directory, file_name)
    return new_file_path


def replace_strange_folder_paths(in_val):
    """Helps make folder path valid"""
    return in_val.replace("\\", "/").replace(" \\", "/").replace(" \\", "/").replace(" /", "/")


def make_filename_valid(filename):
    valid_chars = re.compile(r'[\w.-]+')
    valid_filename = '_'.join(valid_chars.findall(filename))
    invalid_chars = re.compile(r'[^\w.-]+')
    return replace_strange_folder_paths(invalid_chars.sub('_', valid_filename).strip())


def make_filename_valid_from_fullpath(filepath):
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    valid_filename = make_filename_valid(filename)
    return os.path.join(dirname, valid_filename)


def is_filepath_with_filename(filepath):
    file_extensions = ['.jpg', '.jpeg', '.png']
    for ext in file_extensions:
        if filepath.lower().endswith(ext):
            return True
    return False


def make_filepath_valid(filepath):
    # realised it was causing an issue sometimes as filepath had the filename now this should fix it.
    if is_filepath_with_filename(filepath):
        direct, file = os.path.split(filepath)
    else:
        direct = filepath
        file = ""

    if ':' in direct:
        drive, rest = direct.split(':', 1)
        rest = re.sub(r'[^\w \\/]', '', rest).replace("  ", "-")
        out_path = replace_strange_folder_paths((drive + ':' + rest).strip())
    else:
        rest = re.sub(r'[^\w \\/]', '', direct).replace("  ", "-")
        out_path = replace_strange_folder_paths(rest.strip())

    out_path = os.path.join(out_path, file)
    return make_filename_valid_from_fullpath(out_path)


def is_image_match(fl, fl1, hash=None):
    """
    Checks if two images are visually similar (i.e a match)

    :param fl: file path
    :param fl1: file path
    :return: BOOL if match
    """
    # Load images
    if not hash:
        img1 = Image.open(fl)
        # Compute perceptual hashes
        hash = imagehash.phash(img1)

    img2 = Image.open(fl1)
    hash2 = imagehash.phash(img2)

    # Compare hashes
    hamming_distance = hash - hash2

    # Set a threshold for the hamming distance
    if hamming_distance < 10:
        return True, hash
    else:
        return False, hash


def remove_partial_folders_add_match(path):
    return path.replace("partial/", "match/").replace("partial_other/", "match/").replace("folder_match/", "match/")


def closest_matches(input_str, string_series, n=8):
    """returns closest product names"""
    distances = string_series.apply(lambda x: distance(input_str, x))
    closest_matches = distances[distances > 0].sort_values().head(n)
    return string_series.loc[closest_matches.index]


def get_path_after_val(path, val="match"):
    head, tail = os.path.split(path)
    paths = []
    # Split the head until there's only one directory left
    while head and tail:
        paths.append(tail)
        head, tail = os.path.split(head)
    paths.reverse()

    new_path = ""
    for x in paths:
        new_path += "/" + x
        if paths[paths.index(x) - 1] == val:
            break
    return new_path


def remove_alt_from_path(path):
    """remove alt from path"""
    dest, file = os.path.split(path)
    ext = file.split(".")[-1]
    ref = file.replace("." + ext, "")
    split_ref = ref.split("_alt")[0]
    return os.path.join(dest, split_ref + "." + ext)


def remove_partial_folders_add_to_first_folder_change_ref(path, ref):
    path = path.replace("partial/", "match/").replace("partial_other/", "match/").replace("folder_match/", "match/")
    path, file = os.path.split(path)
    ext = file.split(".")[-1]

    # Split the path into head and tail
    new_path = get_path_after_val(path, "match")
    new_full_path = make_filepath_valid(os.path.join(new_path, ref + "." + ext))
    return new_full_path, new_path


def get_smallest_window_res():
    """
    Get the smallest monitor resolution

    Returns:
        Tuple of width and height of the smallest monitor
    """
    sizes = []
    # loop through all monitors and add the resolution to sizes list
    for m in get_monitors():
        sizes.append([m.width * m.height, m.width, m.height])
    # get the largest resolution from sizes list
    largest_resolution = max(sizes, key=lambda x: x[0])
    # return width and height of the largest resolution
    return largest_resolution[1:]


def open_path(path):
    """opens path to sort files"""
    if os.name == 'nt':  # Windows
        subprocess.run(['explorer', path], check=True)
    elif os.name == 'posix':  # Linux or macOS
        subprocess.run(['xdg-open', path], check=True)


def open_file(path):
    imageViewerFromCommandLine = {'linux': 'xdg-open',
                                  'win32': 'explorer',
                                  'darwin': 'open'}[sys.platform]
    subprocess.Popen([imageViewerFromCommandLine, path])


def show_folder_window(path):
    """ for opening folders """
    layout = [
        [sg.Text("Please click open, sort any images and then click done", font=("Helvetica", 20), text_color='blue')],
        [sg.Button("Open", font=("Helvetica", 20), button_color=('white', 'green'), key="Open"),
         sg.Button("Done", font=("Helvetica", 20), button_color=('white', 'red'), key="Done")]]

    window = sg.Window("Window Title", layout)

    while True:
        event, values = window.read()
        if event == "Open":
            open_path(path)
        elif event == "Done":
            window.close()
            break
        elif event == WIN_CLOSED:
            break


def resize_and_pad_with_background(image, desired_width, desired_height, background_color):
    original_height, original_width = image.shape[:2]
    original_aspect_ratio = original_width / original_height
    desired_aspect_ratio = desired_width / desired_height

    if original_aspect_ratio > desired_aspect_ratio:
        new_width = desired_width
        new_height = int(desired_width / original_aspect_ratio)
    else:
        new_width = int(desired_height * original_aspect_ratio)
        new_height = desired_height

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    background_color = np.array(background_color)
    background_color = background_color.astype(np.uint8)

    padded_image = np.full((desired_height, desired_width, 3), background_color, dtype=np.uint8)
    start_y = int((desired_height - new_height) / 2)
    start_x = int((desired_width - new_width) / 2)
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width, :] = resized_image

    return padded_image


def resize_image(img, new_width, new_height):
    original_height, original_width, channels = img.shape
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)

    resized = cv2.resize(img, (int(new_width), int(new_height)))
    return resized


def delete_list_of_files(files):
    for file in files:
        os.remove(file)


def rename_list_of_files(files, alt="_alt"):
    for file in files:
        file_name, key = file
        dir, fl = os.path.split(file_name)
        ext = fl.split(".")[-1]
        just_file = fl.replace("." + ext, "")
        rm_alt = just_file.split(alt)[0]
        shutil.move(file_name, os.path.join(dir, f"{rm_alt}__{key}.{ext}"))


def show_real_check(filenames, ref, description, originals):
    # sg.theme('Dark Brown')
    screen_height, screen_width = get_smallest_window_res()
    screen_width = int(screen_width * 0.75)
    screen_height = int(screen_height * 0.5)

    image_width = screen_width / 3.6
    image_height = min(screen_height / ((len(filenames) // 3) + 1), 300)

    layout = [[sg.Text(ref + '\n' + description, font=('Helvetica', 15), justification='center')]]

    pngs = []
    for file in filenames:
        img = resize_and_pad_with_background(resize_image(cv2.imread(file), image_width, image_height),
                                             int(image_width), int(image_height), (255, 255, 255))
        pngs.append(cv2.imencode(".png", img)[1].tobytes())

    dic = {"keep": [], "delete": []}

    def loop_keys():
        keys = [x.split("_")[0] for x in list(values.keys())]
        for key in keys:
            yield key

    def clear_all():
        for key in loop_keys():
            window.Element(f"{key}_text").update("")

    def all_full():
        for key in loop_keys():
            if not values[f"{key}_text"]:
                return False
        return True

    def get_next_number():
        reorder_list = []
        for key in loop_keys():
            if values[f"{key}_text"] not in ["delete", ""]:
                reorder_list.append(int(values[f"{key}_text"]))
        return max(reorder_list) + 1 if reorder_list else 1

    def get_next_number():
        reorder_list = []
        for key in loop_keys():
            if values[f"{key}_text"] not in ["delete", ""]:
                reorder_list.append(int(values[f"{key}_text"]))
        return max(reorder_list) + 1 if reorder_list else 1

    def get_deal_with_all():
        delete_list = []
        reorder_list = []
        for key in loop_keys():
            if values[f"{key}_text"] == "delete":
                delete_list.append(filenames[int(key)])
            else:
                reorder_list.append([filenames[int(key)], values[f"{key}_text"]])

        if not validate_list([x[1] for x in reorder_list]):
            sg.popup("You haven't got the list of priority right check again please", title="Error", button_type=None,
                     auto_close=False, auto_close_duration=None)
        else:
            return delete_list, reorder_list

    def validate_list(strings):
        # convert the strings to integers
        integers = [int(s) for s in strings]

        # sort the integers
        integers.sort()

        # check for duplicates
        if len(integers) != len(set(integers)):
            return False

        # check if the integers are in order, starting from 1
        for i, num in enumerate(integers, start=1):
            if num != i:
                return False

        return True

    img_list = []
    ind = 0
    for fl in filenames:
        img_list.append(sg.Column([[sg.Image(pngs[ind], size=(image_width, image_height))],
                                   [sg.Button('Keep', key=f"{ind}_keep", metadata={"file": fl}),
                                    sg.Button('Delete', key=f"{ind}_delete", metadata={"file": fl}),
                                    sg.Button('View', key=f"{ind}_view", metadata={"file": fl})],
                                   [sg.Text(originals[ind])], [sg.Input(key=f"{ind}_text")]]))

        if len(img_list) == 3:
            layout.append(img_list)
            img_list = []
        ind += 1
    layout.append(img_list)

    layout.append([[sg.HorizontalSeparator()], [sg.Button('Clear', key="clear"), sg.Button('Done', key="leave")]])

    window = sg.Window('Image Viewer', layout, size=(screen_width, screen_height),
                       element_justification='center')

    while True:
        event, values = window.read()
        if event == 'leave':
            if all_full():
                delete_list, reorder_list = get_deal_with_all()
                window.close()
                return delete_list, reorder_list
            else:
                sg.popup("You haven't filled in all the products", title="Error", button_type=None, auto_close=False,
                         auto_close_duration=None)
        elif "clear" in event:
            clear_all()
        elif "keep" in event:
            key_val = event.split("_")[0]
            window.Element(f"{key_val}_text").update(get_next_number())
        elif "delete" in event:
            key_val = event.split("_")[0]
            window.Element(f"{key_val}_text").update("delete")
        elif "view" in event:
            key_val = event.split("_")[0]
            open_file(window.Element(f"{key_val}_view").metadata["file"])

    window.close()


def show_image_window(filename, ref, description, move_func, ref_matches=None, source=None):
    """
    Displays an image window and provides options to delete, keep, view, or rename the image file.

    :param filename: str - the file path of the image to be displayed.
    :param ref: str - the reference identifier for the image.
    :param description: str - a description of the image.
    :param move_func: function - a function that moves the image file to a different location.
    :return: tuple(str, str) - a tuple containing two values:
        - The subpath of the image file after it has been renamed, if it has been renamed.
        - The filename of the image file if it has been deleted, otherwise None.
    """

    # Read the image
    img = cv2.imread(filename)

    # Resize the window to 50% of the screen size
    screen_height, screen_width = get_smallest_window_res()
    window_height = int(screen_height // 2)
    window_width = int(screen_width // 2)

    has_ref = not ref_matches is None

    # Get the height of the text and buttons
    text_height = 60 + (20 if has_ref else 0) + (15 * len(source) if source else 0)
    button_height = 70

    # Resize the image to fit in the window
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height
    if aspect_ratio > window_width / (window_height - text_height - button_height):
        img_width = window_width
        img_height = int((window_height - text_height - button_height) / aspect_ratio)
    else:
        img_height = window_height - text_height - button_height
        img_width = int(img_height * aspect_ratio)

    # limit the size of the image if it exceeds the size of the window
    if img_width > window_width:
        img_width = window_width
        img_height = int((window_height - text_height - button_height) / aspect_ratio)
    if img_height > window_height - text_height - button_height:
        img_height = window_height - text_height - button_height
        img_width = int(img_height * aspect_ratio)

    img = cv2.resize(img, (int(img_width), int(img_height)))

    # Create the window layout
    layout = [[sg.Text(ref, font=("Helvetica", 20), text_color='blue')],
              [sg.Text(description, font=("Helvetica", 20), text_color='blue')]]

    # if a source file has been passed then show it on the screen so we know where it came from
    if source:
        for i, so in enumerate(source):
            layout.append([sg.Text("Original File:", font=("Helvetica", 12), text_color='black'),
                           sg.Text(so, font=("Helvetica", 12), text_color='black')])

    layout += [[sg.Button("delete", font=("Helvetica", 20), button_color=('white', 'red')),
                sg.Button("keep", font=("Helvetica", 20), button_color=('white', 'green')),
                sg.Button("view", font=("Helvetica", 20), button_color=("black", "yellow")),
                sg.Button("rename", font=("Helvetica", 20), button_color=("black", "orange")),
                ], [sg.Text('Actual SKU:'), sg.Input(key='input')]]

    if has_ref:
        """adds a dropdown if needed"""
        layout.append([sg.Text('Choose an SKU:'), sg.Combo(ref_matches, size=(20, 1), key="combo"),
                       sg.Checkbox("Use this?", key="check")])

    layout.append([sg.Image(data=cv2.imencode(".png", img)[1].tobytes(), key="IMAGE")])

    # Create the window
    window = sg.Window("Image Window", layout, size=(int(window_width), int(window_height)))

    # Event loop
    while True:
        event, values = window.read()
        if event == "delete":

            window.close()
            return None, filename

        elif event == "keep":
            # Move the file to the desired location
            move_func(filename, remove_partial_folders_add_match(filename), ref, move=True)
            window.close()
            break
        elif event == "view":
            # Move the file to the desired location
            open_file(filename)

        elif event == "rename":
            # Move the file to the desired location
            if (len(values["input"]) == 0 and not values["check"]) or (values["check"] and not values["combo"]):
                sg.popup("You haven't given a sku", title="Error", button_type=None, auto_close=False,
                         auto_close_duration=None)
            else:
                new_code = values["combo"] if values["check"] else values["input"]
                new_path, sub_path = remove_partial_folders_add_to_first_folder_change_ref(filename, new_code)
                move_func(filename, new_path, ref, move=True)

                window.close()
                return sub_path, None

        elif event == WIN_CLOSED:
            break
    return None, None


def is_file_same(file1, file2):
    """checks if two files are the same """
    # Open the files in binary mode
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        # Calculate the md5 hash of each file
        hash1 = hashlib.sha1(f1.read()).hexdigest()
        hash2 = hashlib.sha1(f2.read()).hexdigest()
    return hash1 == hash2


def silence_output(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return result

    return wrapper


class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush();
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush();
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def add_suffix_to_file(file_path, path="alt", random=True):
    """
    adds a suffix to a file name, preserving the ext
    :param file_path:
    :param path:
    :param random:
    :return:
    """
    base, ext = os.path.splitext(file_path)
    suffix = f"_{path}" + (f"{random.randint(0, 99999)}" if True else "")
    return (base + suffix + ext).strip()


def make_folders_from_filepath(filepath):
    """
    Makes all missing folders in the filepath
    :param filepath: str
    """
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)


def shrink_stock_ref(nm):
    """Translates the stock reference into something that could be readable"""
    return str(nm).lower().translate(str.maketrans("", "", " -_.\\/"))


def change_filename(nm):
    """here for compatibility"""
    return shrink_stock_ref(nm)


def count_matches(strings, x, path):
    """
    Takes an array and a string, also the path name of the file. Then checks for 4 match types

    Full Match - any string matches
    Partial match - any String in arrString[:len(inputstring)] == inputstring
    Reverse Partial Match - any String in inputstring[:len(arrString)] == arrString
    Path Match - does the path match the input string - i.e /test/path/8820/front.jpg   == 8820  - True as the path matches the string in top folder
    :param strings:
    :param x:
    :param path:
    :return:
    """
    exact_matches = strings[np.where(strings == x)[0]]

    if len(x) >= 3:
        matches = np.char.startswith(strings, x)
        first_section_matches = strings[np.where(matches)[0]]
    else:
        first_section_matches = np.array([])

    max_len = np.max([len(s) for s in strings])
    repeated_search_term = x.rjust(max_len)
    out = np.array([repeated_search_term[:len(s)] for s in strings])
    partial_matches = strings[np.where(out == strings)[0]]

    # Create a NumPy array of strings
    search_prefix = x[:4]
    # Find all occurrences where the first four characters match the search prefix
    first_part_matches = strings[np.where(np.char.find(strings, search_prefix) == 0)[0]]

    path, file = os.path.split(path)
    split_path = path.split("/")[-1].split("\\")

    path_sp = make_filename_valid(shrink_stock_ref(split_path[-1]))
    path_matches = strings[np.where(strings == path_sp)[0]]

    path_matches_one = []
    if len(split_path) > 1:
        path_sp_one = make_filename_valid(shrink_stock_ref(split_path[-2]))
        path_matches_one = strings[np.where(strings == path_sp_one)[0]]

    def closest_match(strings, search_term):
        if len(strings) > 0:
            distances = np.array([distance(string, search_term) for string in strings])
            closest_index = np.argmin(distances)
            return strings[closest_index]
        return None

    return closest_match(exact_matches, x), \
        first_section_matches.tolist(), \
        partial_matches.tolist(), \
        closest_match(path_matches, x) or closest_match(path_matches_one, x), \
        first_part_matches.tolist()


def pprint(text, i=None, total=None, start=None):
    nnow = datetime.datetime.now()

    print_str = ""
    if start:
        diff = int((nnow - start).total_seconds())
        hours = diff // 3600
        minutes = (diff % 3600) // 60
        secs = diff % 60
        print_str += "Run Time: " + "{:02d}:{:02d}:{:02d}  ".format(hours, minutes, secs)

    if i and total:
        per = diff / i
        secs = int(per * total)
        seconds = secs - diff

        hours_left = seconds // 3600
        minutes_left = (seconds % 3600) // 60
        secs_left = seconds % 60
        print_str += "Time left: " + "{:02d}:{:02d}:{:02d} ".format(hours_left, minutes_left,
                                                                    secs_left) + f" no {i}/{total} "

    print_str += "" if not start or not (i and total) else "  -  "
    print(
        "                                                                                                                                                        ",
        end="\r")
    print(print_str + text, end="\r")


def is_large_file(file):
    """
    Check if the file is ok to load into the system

    :param file: path
    :return:
    """
    return 10240 < file.stat().st_size < 10485760 * 1.5


def shrink_stock_ref(ref):
    return str(ref).lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "").replace(" /",
                                                                                                        "/").replace(
        "\\", "").replace("  ", "").strip()


def fix_nan(row):
    for i, row_item in enumerate(row):
        row[i] = convert_nan(row[i])
    return row


def get_name_n_ext_from_filename(filename):
    ext = filename.split(".")[-1]
    name = filename.replace("." + ext, "")
    return name, ext


def get_alternate_files(filename, alt="alt"):
    """
    Returns a list of all files that have the same base filename as the provided file,
    with "_alt" appended to the end.

    Args:
        filename (str): The name of the file for which to find alternates.

    Returns:
        list: A list of filenames that match the criteria. The original filename is included
        in the list as the first element.
    """
    # Create a list to store all alternate files
    alternate_files = []

    # Get the directory path of the provided file
    dir_path = os.path.dirname(filename)

    # Get the base filename of the provided file (without the path)
    base_filename = os.path.basename(filename)
    name, ext = get_name_n_ext_from_filename(base_filename)

    # Loop through all files in the directory
    for file in os.listdir(dir_path):
        # Check if the file starts with the base filename + "_{alt}"
        if file.startswith(name + "_" + alt):
            # If it does, add the full file path to the alternate_files list
            alternate_files.append(os.path.join(dir_path, file))

    # Return the original filename plus all alternate files found
    return [filename] + alternate_files
