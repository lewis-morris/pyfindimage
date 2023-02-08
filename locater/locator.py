import datetime
import os
import random
import re
import shutil

import cv2
import imagehash
import numpy as np
import pandas as pd
from Levenshtein import distance
from PIL import Image


class String:
    def change_stock_ref(self):
        nm.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "").replace(" /", "/").strip()


df = pd.read_csv("C:/Users/ChrisColeman/Desktop/colemandata.csv")
dfpro = pd.read_excel("C:/Users/ChrisColeman/Desktop/prostock.xlsx")
df["changedSku"] = df["Code"].apply(change_stock_ref)
dfpro["changedSku"] = dfpro["scode"].apply(change_stock_ref)
dfpron = dfpro["changedSku"].to_numpy().astype(str)
dfpron_look = dfpro[["changedSku", "scode", "supplier"]].to_numpy().astype(str)

matches = 0
partials = 0
dupes = 0
errors = []
found_web = {}
found_pro = {}


def save_dics():
    pd.DataFrame([[v["match"], v["partial"]] for k, v in found_web.items()], index=found_web.keys(),
                 columns=["matches", "partials"]).sort_index().to_excel("C:/Users/ChrisColeman/Desktop/foundweb.xlsx")
    pd.DataFrame([[v["match"], v["partial"]] for k, v in found_pro.items()], index=found_pro.keys(),
                 columns=["matches", "partials"]).sort_index().to_excel("C:/Users/ChrisColeman/Desktop/foundpro.xlsx")


def get_realsku(ref, web=True, sup=False):
    if web:
        ar = dfwebn_look
    else:
        ar = dfpron_look

    search_value = ref
    index = np.where(ar[:, 0] == search_value)
    try:
        if web:
            return ar[index][0][1]
        else:
            return ar[index][0][1:]
    except Exception as e:
        errors.append(["get_realsku", ref, e])


get_realsku(change_stock_ref("10024GOOD"), False)


def convert_nan(val):
    return "" if isinstance(val, float) and np.isnan(val) else str(val)


# def copy_file_with_checks(src_file, dst_file):
#
#     dst_file = dst_file.replace("\\","/").replace(" \\", "/").replace(" \\", "/").replace(" /", "/")
#     make_folders_from_filepath(dst_file)
#
#     try:
#         if os.path.exists(dst_file):
#             dst_file = add_suffix_to_file(dst_file)
#             shutil.copy2(src_file, dst_file)
#         else:
#             shutil.copy2(src_file, dst_file)
#     except:
#         print("Error with " + dst_file)
# def crop_img(file_path):
#     # Load the image
#     image = Image.open(file_path)
#     # Crop the image
#     crop_width = int(image.width * 0.01)
#     crop_height = crop_width
#     cropped_image = image.crop((0, 0, crop_width, crop_height))
#     return cropped_image
#
# def is_product(img):
#     # Convert the image to RGBA format
#
#     # Get the white color range
#     white_range = (230, 230, 230, 255)
#     # Get the pixel data as a numpy array
#     pixel_data = np.array(img)
#     # Check the number of white pixels
#     white_pixels = np.sum(np.all(pixel_data == white_range, axis=-1))
#     #get pixels
#     total_pixels = img.width * img.height
#     # Calculate the percentage of white and transparent pixels
#     white_percentage = white_pixels / total_pixels
#     # Return True if the image is primarily off-white or transparent
#     return white_percentage < 0.9

def is_image_product(file_path):
    # Load the image
    image = cv2.imread(file_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Increase the contrast by 50%
    image = np.interp(image, (0, 255), (-0.5, 0.5))
    image = np.clip(image * 1.5 + 0.5, 0, 255).astype(np.uint8)
    # Get the top left and bottom right pixels of the image
    top_left = image[0, 0]
    bottom_right = image[-1, -1]
    # Check if the top left and bottom right pixels are either white or transparent
    is_top_left_white_or_transparent = np.all(top_left >= (230, 230, 230)) or np.all(top_left == (0, 0, 0))
    is_bottom_right_white_or_transparent = np.all(bottom_right >= (230, 230, 230)) or np.all(bottom_right == (0, 0, 0))
    # Return True if both the top left and bottom right pixels are either white or transparent
    return is_top_left_white_or_transparent and is_bottom_right_white_or_transparent


def add_background_directory(file_path, product_ref):
    """
    adds a new top level directory to the path, just before the filename

    :param file_path: string file path
    :param product_ref: new folder name
    :return: reconstructed file path
    """
    directory, file_name = os.path.split(file_path)
    new_directory = make_filepath_valid(os.path.join(directory, product_ref + "_background")).replace("\\",
                                                                                                      "/").replace(
        " \\", "/").replace(" \\", "/").replace(" /", "/")
    new_file_path = os.path.join(new_directory, file_name)
    return new_file_path


def copy_file_with_checks(src_file: str, dst_file: str, ref: str) -> None:
    """
    Copies a file to a new path, if the imege is
    :param src_file: Source File to copy
    :param dst_file: Destination file path
    :param ref: Stock ref
    """
    global dupes
    dst_file = dst_file.replace("\\", "/").replace(" \\", "/").replace(" \\", "/").replace(" /", "/")
    make_folders_from_filepath(dst_file)

    if is_image_product(file_path):
        # if the image is a product then amend dst folder
        dst_file = add_background_directory(dst_file, ref)
    # make folders in path if not exist
    make_folders_from_filepath

    try:
        # if file exists then  - check if the image matches the original, if match then append 'same' to the file
        # name, if not image doesnt match and add alt
        if os.path.exists(dst_file):
            if is_image_match(src_file, dst_file):
                dupes += 1
                dst_file = add_suffix_to_file(dst_file, "same")
            else:
                dst_file = add_suffix_to_file(dst_file, "alt")
        shutil.copy2(src_file, dst_file)

    except Exception as e:
        errors.append(["copy file", ref, e])
        print("Error with " + dst_file)


def make_folders_from_filepath(filepath):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)


def add_suffix_to_file(file_path, path="alt"):
    base, ext = os.path.splitext(file_path)
    suffix = f"_{path}_{random.randint(0, 99999)}"
    return (base + suffix + ext).strip()


def change_stock_ref(nm):
    return nm.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "").replace(" /", "/").strip()


def change_filename(nm):
    return nm.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "").replace(" /", "/").replace(
        " \\", "/").strip()


def get_web_path(row, ref, typ, web):
    base = get_base(web)
    paths = [convert_nan(row["Supplier"].to_list()[0]), typ, convert_nan(row["Range"].to_list()[0]),
             convert_nan(row["Type"].to_list()[0]), ref]
    path = os.path.join(base, *[x for x in paths if x])

    return path.replace("//", "/").strip().replace(" /", "/").replace(" /", "/")


def get_path(ref, typ, sup, web):
    base = get_base(web)
    path = os.path.join(base, sup, typ, ref)
    return path.replace("//", "/").strip().replace(" /", "/").replace(" /", "/")


def make_filename_valid(filename):
    valid_chars = re.compile(r'[\w.-]+')
    valid_filename = '_'.join(valid_chars.findall(filename))
    invalid_chars = re.compile(r'[^\w.-]+')
    return invalid_chars.sub('_', valid_filename).strip()


def make_filepath_valid(filepath):
    drive, rest = (filepath.split(':', 1) + [''])[:2]
    rest = re.sub(r'[^\w \\/]', '', rest).replace("  ", "-")
    return (drive + ':' + rest if drive else rest).strip()


def is_image_match(fl, fl1):
    # Load images
    img1 = Image.open(fl)
    img2 = Image.open(fl1)

    # Compute perceptual hashes
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    # Compare hashes
    hamming_distance = hash1 - hash2

    # Set a threshold for the hamming distance
    if hamming_distance < 10:
        return True
    else:
        return False


def count_matches(strings, x, path):
    exact_matches = strings[np.where(strings == x)[0]]

    matches = np.char.startswith(strings, x)
    first_section_matches = strings[np.where(matches)[0]]

    max_len = np.max([len(s) for s in strings])
    repeated_search_term = x.rjust(max_len)
    out = np.array([repeated_search_term[:len(s)] for s in strings])
    partial_matches = strings[np.where(out == strings)[0]]

    path, file = os.path.split(path)
    path_sp = make_filename_valid(change_stock_ref(path.split("/")[-1].split("\\")[-1]))

    path_matches = strings[np.where(strings == path_sp)[0]]

    def closest_match(strings, search_term):
        if len(strings) > 0:
            distances = np.array([distance(string, search_term) for string in strings])
            closest_index = np.argmin(distances)
            return strings[closest_index]
        return None

    return closest_match(exact_matches, x), closest_match(first_section_matches, x), closest_match(partial_matches,
                                                                                                   x), closest_match(
        path_matches, x)


def check_if_match(ref):
    ref = ref.replace("." + ref.split(".")[-1], "")
    ref = change_filename(ref)

    # dfweb
    # dfpro
    # found

    for i in range(len(df)):
        row = df.iloc[i]
        code = row["Code"]
        code = change_filename(code)

        if code == ref:
            return row, True
            df.iloc[i]["found"] = True
        if code in ref or ref in code:
            if not df.iloc[i]["found"] is True:
                df.iloc[i]["found"] = "maybe"
            return row, False
    return False, False


root_dir = 'Y:/'
all_files = []

now = datetime.datetime.now()


def pprint(text, i=None, total=None):
    nnow = datetime.datetime.now()
    diff = int((nnow - now).total_seconds())
    hours = diff // 3600
    minutes = (diff % 3600) // 60
    secs = diff % 60

    if i and total:
        per = diff / i
        seconds = int(per * total)
        hours_left = seconds // 3600
        minutes_left = (seconds % 3600) // 60
        secs_left = seconds % 60
        print("Run Time: " + "{:02d}:{:02d}:{:02d}".format(hours, minutes,
                                                           secs) + " Time left: " + "{:02d}:{:02d}:{:02d}".format(
            hours_left, minutes_left, secs_left) + f" no {i}/{total} - " + text)
        return

    print("Run Time: " + "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs) + "  - " + text)
    return diff


def is_large_file(file):
    return 10240 < file.stat().st_size < 10485760 * 1.5


def loop_through_directories(path):
    for entry in os.scandir(path):
        found = sum([1 if x in entry.path else 0 for x in
                     ["recyc", "Delivery Notes", "Pick No", "Invoices", ".ipynb_checkpoints"]])
        if found == 0:
            if entry.is_dir():
                loop_through_directories(entry.path)
            else:
                file = entry.name
                root = entry.path.replace(entry.name, "").replace("\\", "/")
                file_path = os.path.join(root, file)
                image = 1 if file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif", "webp"] else 0
                if image and is_large_file(entry):
                    all_files.append([file_path, root, file])

                if len(all_files) % 10000 == 0:
                    pprint("Found len" + str(len(all_files)))


loop_through_directories(root_dir)


#
# for root, dirs, files in os.walk(root_dir):
#     found = sum([1 if x in root else 0 for x in ["recyc", "Invoices"]])
#     if found == 0:
#         for file in files:
#
#
def get_base(web=True):
    return "C:/Users/ChrisColeman/Desktop/" + ("web" if web else "pro") + "/"


def add_to_dict(sku, match=True, web=True):
    di = found_web if web else found_pro
    if not sku in di:
        di[sku] = {"match": 0, "partial": 0}
    di[sku]["partial"] += 1 if not match else 0
    di[sku]["match"] += 1 if match else 0


def deal_with_match(matchqty, file, filebase, ref, web=True, row=None):
    global matches
    global partials

    ext = file.split(".")[-1]

    if matchqty[0]:
        matches += 1
        typ = "match/"
    elif matchqty[1]:
        typ = "partial/"
        partials += 1
    elif matchqty[2]:
        typ = "partial_other/"
        partials += 1
    elif matchqty[3]:
        typ = "folder_match/"
        partials += 1

    try:
        if web:
            sku_real = get_realsku(ref, True)
            filename = make_filename_valid(sku_real) + "." + ext
            row = df[df["Code"] == sku_real]
            new_path = get_web_path(row, filename, typ, web)
        else:
            out = get_realsku(ref, False)
            filename = make_filename_valid(out[0]) + "." + ext
            sku_real = out[0]
            sup = out[1]
            new_path = get_path(filename, typ, sup, web)

        add_to_dict(sku_real, typ == "match/", web)

        copy_file_with_checks(os.path.join(filebase + file), new_path, sku_real)
    except Exception as e:
        errors.append(["deal_with_match", ref, e])


now = datetime.datetime.now()
files_len = len(all_files)

for i, listed_file in enumerate(all_files):
    file_path, root, file = listed_file

    code = change_stock_ref(file.replace(file.split(".")[-1], ""))
    pro_match = count_matches(dfpron, code, file_path)
    check_pro_match = [x for x in list(pro_match) if x]

    if len(check_pro_match) > 0:
        deal_with_match(pro_match, file, root, check_pro_match[0], False)

    web_match = count_matches(dfwebn, code, file_path)
    check_web_match = [x for x in list(web_match) if x]
    if len(check_web_match) > 0:
        deal_with_match(web_match, file, root, check_web_match[0], True)

    if i % 1000 == 0:
        save_dics()
        pprint(f"Img {i} checked   Total Matches: {matches}  Total Partials {partials}  Total Dupes {dupes}", i,
               files_len)

save_dics()
print(errors)
errors
