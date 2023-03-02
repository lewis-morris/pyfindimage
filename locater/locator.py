import datetime
import filecmp
import os
import pickle
import random
import re
import shutil

import cv2
import imagehash
import numpy as np
import pandas as pd
from Levenshtein import distance
from PIL import Image

from locater.functions import is_large_file, pprint, shrink_stock_ref, fix_nan, make_filename_valid, \
    count_matches, add_suffix_to_file, is_image_match, make_folders_from_filepath, \
    add_top_level_parent_directory, is_image_product, replace_strange_folder_paths, get_alternate_files, \
    show_image_window, show_folder_window, remove_alt_from_path, make_filepath_valid, closest_matches, is_file_same, \
    remove_partial_folders_add_match, show_real_check, delete_list_of_files, rename_list_of_files


class Locator:
    def __init__(self, job_name, search_path: str, out_path: str, data_file: str, *args, **kwargs):
        """Class for locating and organizing files based on search terms.

        :param job_name: Name of job
        :param search_path: Path to search for files.
        :param out_path: Path to output files.
        :param data_file: File consisting of columns in this order: search_term, description, groupingmain, groupingsecondary, etc.
        """

        # source of the files and its length so it doesn't have to be computed each time
        self.job_name = job_name
        self.files_source = []
        self.files_len = 0
        # list of eventual files once loaded
        self.search_path = search_path  # path to search
        self.out_path = out_path  # output path
        # pandas df of all the data
        self._base_data = self._load_data(data_file)
        self._out_data = self._create_out_data()
        # numpy array to perform matching on
        self.code_data = self._base_data["changedSku"].to_numpy().astype(str)
        # data for getting path etc
        self.lookup_data = self._base_data.to_numpy().astype(str)
        # should print info
        self.verbose = True if "verbose" not in kwargs else True if kwargs["verbose"] else False
        # extra information that could be used later
        self.extra = {"exact_match_paths": [], "search_matches": 0, "search_partials": 0, "search_dupes": 0,
                      "matches_ext": "same", "alt_ext": "alt", "current_match_val": 0}
        # list of the new file paths
        self.paths = {"matches": [], "partials": []}
        # dict of all copies key is new_location
        self.file_copies = {}
        self.hashes = {}

    def save_job(self):
        filename = f"{self.job_name}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_from_save(self):
        filename = f"{self.job_name}.pickle"
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

    def _create_out_data(self):
        df = self._base_data.copy()
        df = df[[df.columns.to_list()[0], df.columns.to_list()[2]]]
        df["matches"] = False
        df["partials"] = False
        return df

    def _load_data(self, file_path):
        # Determine the file extension
        file_extension = file_path[-5:]
        # Choose the appropriate reader function based on the file extension
        if "xls" in file_extension:
            reader = pd.read_excel
        if "xlsx" in file_extension:
            reader = pd.read_excel
        elif "csv" in file_extension in "csv":
            reader = pd.read_csv

        assert reader, "filetype not supported, please use xls, xlsx, or csv"

        # Load the data into a Pandas DataFrame
        df = reader(file_path)

        # Drop duplicate values in the first column of the DataFrame
        df.drop_duplicates(subset=df.columns[0], inplace=True)

        # Create a new column "changedSku" using the `shrink_stock_ref` function
        df["changedSku"] = df.iloc[:, 0].apply(shrink_stock_ref)

        # Reorder the columns of the DataFrame
        columns = df.columns.tolist()
        new_col_order = [columns[0], "changedSku"] + columns[1:-1]
        df = df[new_col_order]

        # Apply the `fix_nan` function to fill missing values
        df = df.apply(fix_nan, axis=1)

        return df

    def _get_alt_path(self):
        return self.extra["alt_ext"] or "_alt"

    def _get_match_path(self):
        return self.extra["matches_ext"] or "_same"

    def _get_matches(self):
        return self.extra["search_matches"]

    def _get_partials(self):
        return self.extra["search_partials"]

    def _get_dupes(self):
        return self.extra["search_dupes"]

    def _add_match(self, ref=None):
        self.extra["search_matches"] += 1
        if ref:
            self._out_data.loc[self._out_data[self._out_data.columns.tolist()[0]] == ref, "matches"] = True

    def _add_partial(self, ref=None):
        self.extra["search_partials"] += 1
        if ref:
            if type(ref) is str:
                self._out_data.loc[self._out_data[self._out_data.columns.tolist()[0]] == ref, "partials"] = True
            else:
                for re in ref:
                    self._out_data.loc[self._out_data[self._out_data.columns.tolist()[0]] == re, "partials"] = True

    def _add_dupe(self):
        self.extra["search_dupes"] += 1

    def _set_path(self, ref, path):

        if "/match/" in path:
            self.paths["matches"].append([ref, path])
        else:
            self.paths["partials"].append([ref, path])

    def _get_desc(self, ref):
        """Return the first matching description for the given reference.

        :param ref: The reference to search for.
        :return: The first matching description for the reference.
        """
        return self.lookup_data[np.where(self.lookup_data[:, 0] == ref)[0], 2][0]

    def _get_row(self, ref):
        """Return the first matching row for the given reference.

        :param ref: The reference to search for.
        :return: The first matching row for the reference.
        """
        return self.lookup_data[np.where(self.lookup_data[:, 0] == ref)[0]][0, 3:]

    def _get_path_end(self, ref):
        """Return the path constructed from the attributes in the data.

        :param ref: The reference to use for finding the attributes.
        :return: The path constructed from the attributes.
        """
        return os.path.join(*self._get_row(ref).tolist())

    def _ref_to_shrunk(self, ref):
        """Return the shrunken reference for the given reference.

        :param ref: The reference to shrink.
        :return: The shrunken reference.
        """
        return self.lookup_data[np.where(self.lookup_data[:, 0] == ref)[0], 1][0]

    def _shrunk_to_ref(self, shrunk):
        """Return the original reference for the given shrunken reference.

        :param shrunk: The shrunken reference to expand. Can be a list, if so, a list of results is returned
        :return: The original reference.
        """
        if isinstance(shrunk, str):
            return self.lookup_data[np.where(self.lookup_data[:, 1] == shrunk)[0], 0][0]
        else:
            return [self.lookup_data[np.where(self.lookup_data[:, 1] == s)[0], 0][0] for s in shrunk]

    def _get_hash(self, src):
        hash = None
        if src in self.hashes:
            hash = self.hashes[src]
        return hash
    def _deal_with_matches_etc(self, dst_file, src_file, copymv_func=shutil.copy2):
        """
        Handles files with matching names in the same directory as the input file. Checks to see if any match and then keeps the largest one.

        :param filename: The input file name.
        """
        # Get the directory of the input file
        directory = os.path.dirname(dst_file)
        filename = os.path.basename(dst_file)
        ext = filename.split(".")[-1]
        ref = filename.replace("." + ext, "")

        # Get all the filenames in the directory
        filenames = get_alternate_files(dst_file, self._get_alt_path())

        for check_file in filenames:
            # Check if the filename matches the input file name or follows the pattern "{filename}_alt*"

            i_match, hash_out = is_image_match(src_file, check_file, self._get_hash(self.file_copies.get(check_file)))
            self.hashes[src_file] = hash_out

            if i_match:  # filecmp.cmp(src_file, check_file):
                # If a matching filename is found, get the full file path
                matching_file = os.path.join(directory, check_file)
                # Check if the matching file is an image match with the input file
                # If the matching file is an image match, compare the file sizes
                if os.path.getsize(check_file) > os.path.getsize(check_file):
                    # If the input file is larger, delete the matching file and keep the input file
                    os.remove(matching_file)
                    copymv_func(src_file, matching_file)
                    self.file_copies[matching_file] = src_file
                # End the loop once a match has been found and handled
                return matching_file

        new_filename = f"{ref}_{self._get_alt_path()}" + str(random.randint(0, 99999)) + "." + ext
        new_full_path = os.path.join(directory, new_filename)
        copymv_func(src_file, new_full_path)
        self.file_copies[new_full_path] = src_file
        return new_full_path

    def _copy_file_with_checks(self, src_file: str, dst_file: str, ref: str, move=False) -> None:
        """
        Copies a file to a new path while checking for duplicates.

        :param src_file: The source file to be copied.
        :param dst_file: The destination file path.
        :param ref: The reference string used in naming conventions.
        :param move: If true then the default operation is copy, else move
        """

        if move:
            copymv_func = shutil.move
        else:
            copymv_func = shutil.copy2

        # Replace strange folder paths in the destination file path
        dst_file = replace_strange_folder_paths(dst_file)

        # Remove alt from path if its there
        dst_file = remove_alt_from_path(dst_file)

        # If the image is a product, amend the destination folder
        if is_image_product(src_file):
            dst_file = add_top_level_parent_directory(dst_file, ref, "_background")

        # fixes filepath if needed
        dst_file = make_filepath_valid(dst_file)

        # Make folders in the file path if they do not exist
        make_folders_from_filepath(dst_file)

        try:
            # If the destination file already exists, check if the image matches the original.
            # If it matches, add keep the larger and remove the smaller. If it doesn't match, add a '_alt' suffix and saver.
            if os.path.exists(dst_file):
                dst_file = self._deal_with_matches_etc(dst_file, src_file, copymv_func)
            else:
                # Copy the source file to the destination file
                copymv_func(src_file, dst_file)

            if dst_file:
                self._set_path(ref, dst_file)
                self.file_copies[dst_file] = src_file
                return dst_file


        except Exception as e:
            # Catch any errors that occur during the copying process
            print("Error with " + dst_file)

    def _rename_largest_image(self, filepath: str) -> None:
        # Get the directory and filename without the suffix
        directory, filename = os.path.split(filepath)
        base, ext = os.path.splitext(filename)

        # Find all files with the same base name and the "_same_" suffix
        same_files = [f for f in os.listdir(directory) if f.startswith(base + "_same_")]

        # If there are no such files, return immediately
        if not same_files:
            return

        # Find the file with the largest size
        largest_file = max(same_files, key=lambda f: Image.open(os.path.join(directory, f)).size[0] *
                                                     Image.open(os.path.join(directory, f)).size[1])

        # Rename the largest file to the original name
        os.rename(os.path.join(directory, largest_file), os.path.join(directory, base + ext))

        # Delete the rest of the files
        for f in same_files:
            if f != largest_file:
                os.remove(os.path.join(directory, f))

    def _deal_with_match(self, matchqty, file, filebase, ref):
        """Handle the processing of matched or partially matched files

        :param matchqty: match quantity information
        :param file: file name to be processed
        :param filebase: base file path
        :param ref: reference information

        :return: None
        """
        ext = file.split(".")[-1]
        # convert shunk to real sku
        sku_real_ = self._shrunk_to_ref(ref)

        partial = True
        if matchqty[0]:
            self._add_match(sku_real_)
            typ = "match/"
            partial = False
        elif matchqty[1]:
            typ = "partial/"
            self._add_partial(sku_real_)
        elif matchqty[2]:
            typ = "partial_other/"
            self._add_partial(sku_real_)
        elif matchqty[3]:
            typ = "folder_match/"
            self._add_partial(sku_real_)
        elif matchqty[4]:
            typ = "start_match/"
            self._add_partial(sku_real_)

        try:
            if isinstance(sku_real_, str):
                sku_real_ = [sku_real_]

            for sku_real in sku_real_:
                # get filename
                new_filename = make_filename_valid(sku_real) + "." + ext
                # get new path
                new_path = os.path.join(self.out_path, typ, self._get_path_end(sku_real), new_filename)
                # get source
                source_fl = os.path.join(filebase + file)
                # check not in match folder already

                copy = True
                if partial:
                    copy = not self._check_match(source_fl, new_path, sku_real, ref)
                # copy file
                copy and self._copy_file_with_checks(source_fl, new_path, sku_real)

        except Exception as e:
            pass
            # errors.append(["deal_with_match", ref, e])

    def _check_match(self, src, dest, sku_real, ref):
        """
        Checks to see if there is already a match in the MATCH folder, if so there is no reason
        to add this file to the partials folder
        """
        match_dest = remove_partial_folders_add_match(dest)
        try:
            files = get_alternate_files(match_dest, self._get_alt_path())
            for file in files:
                try:
                    if is_image_match(file, src):
                        return True
                    if filecmp.cmp(file, src):
                        return True
                except:
                    pass
        except:
            pass
        return False

    def _check_paths(self, path):
        """
        Checks if `path` and `self.out_path` are provided, and if `self._base_data` is a `pd.DataFrame`.

        :param path: the path to search for files. If not provided, defaults to `self.search_path`.
        :type path: str, optional

        :raises AssertionError: if `path` is not provided, `self.out_path` is not provided, or `self._base_data` is not a `pd.DataFrame`.
        """
        assert path, "Path must be provided"
        assert self.out_path, "Output path must be provided"
        assert isinstance(self._base_data, pd.DataFrame), "Base data must be a Pandas DataFrame"

    def _get_files(self, path, arr=None):
        """
        Iterates over all entries in the given `path` and returns information about the files that are images and not too large.

        :param path: the path to search for files.
        :type path: str

        :return: a list of lists, where each inner list contains information about a file.
        :rtype: list
        """

        if arr is None:
            arr = self.files_source

        for entry in os.scandir(path):
            if not any(
                    x in entry.path for x in ["recyc", "Delivery Notes", "Pick No", "Invoices", ".ipynb_checkpoints"]):
                if entry.is_dir():
                    self._get_files(entry.path, arr)
                else:
                    file = entry.name
                    root = entry.path.replace(entry.name, "").replace("\\", "/")
                    file_path = os.path.join(root, file)
                    image = 1 if file.split(".")[-1].lower() in ["jpg", "jpeg", "png"] else 0
                    if image and is_large_file(entry):
                        arr.append([file_path, root, file])

                    if self.verbose and len(arr) % 10000 == 0:
                        pprint("Found len" + str(len(arr)))

        return arr

    def _build_file_lists(self, frm="partials"):

        """gets list of all files that were partial matches
        :param frm: either "partials" or "matches" log
        :return list of paths, refs, and descriptions
        """
        base_paths = []
        for fl in self.paths[frm]:
            ref, path = fl
            alternate_files = get_alternate_files(path, self._get_alt_path())
            source_files = [self.file_copies[x] for x in alternate_files]
            base_paths.append([ref, self._get_desc(ref), alternate_files, source_files])
        return base_paths

    def load_files(self, path=None):
        """
        Loads the file into the object by searching the file path for all images and storing their paths and related information.

        :param path: the path to search for files. If not provided, defaults to `self.search_path`.
        :type path: str, optional

        :return: None
        """
        if not path:
            path = self.search_path

        self._check_paths(path)
        self._get_files(path)
        self.files_len = len(self.files_source)
        if self.verbose:
            print("\nLoad files complete")

    def match_files(self):
        """
        Actually attempts to match the files, reads the file list, and then searches the list of files,
        checks for matches, and then if found copies the file to the new location

        :return:
        """

        now = datetime.datetime.now()
        try:
            for i, listed_file in enumerate(self.files_source[self.extra["current_match_val"]:]):

                # this is used so you can re-load from where you left off
                i = self.extra["current_match_val"]
                self.extra["current_match_val"] += 1

                # splits file into its parts
                file_path, root, file = listed_file
                # shinks the file name with the same rules as the input data does
                code = shrink_stock_ref(file.replace(file.split(".")[-1], ""))
                # checks for a match of either, full, partial, reverse partial, or directory match
                match = count_matches(self.code_data, code, file_path)
                # gets the match

                check_match = [x for x in list(match) if x]

                # deals with it if one found
                if len(check_match) > 0:
                    if isinstance(check_match[0], list):
                        # this is here becuase sometimes theres more than once match
                        for mtch in check_match[0]:
                            self._deal_with_match(match, file, root, mtch)
                    else:
                        self._deal_with_match(match, file, root, check_match[0])
                # prints if needed
                if self.verbose and i % 50 == 0:
                    pprint(
                        f"Img {i} checked   Total Matches: {self._get_matches()}  Total Partials {self._get_partials()}  Total Dupes {self._get_dupes()}",
                        i,
                        self.files_len, now)
                if i % 200 == 0:
                    self.save_job()

            self.save_job()

            if self.verbose:
                print("\nMatch files complete")
        except:
            pass

    def check_move_partials(self):
        """Shows all images that are in the partial folders, it then shows you the desctiption the product reference
         and gives the user the option to move to correct folder or to delete them.
        """
        files = self._build_file_lists()

        # list of files that need to be deleted later
        delete_list = []
        # these paths need to have images allocated
        paths_to_check_move = []
        for ref, desc, files, source in files:
            for path in files:
                # this offers some other products it may be
                matches = closest_matches(ref, self._base_data.iloc[:, 0]).values.tolist()
                #checks if the file still exists
                if os.path.isfile(path):
                    # shows the window to choose what to do with partial match
                    out_path, delete_path = show_image_window(path, ref, desc, self._copy_file_with_checks, matches, source)
                    # adds output to list for later processing
                    if out_path:
                        paths_to_check_move.append(out_path)
                    if delete_list:
                        delete_list.append(delete_path)
                else:
                    if self.verbose:
                        print(f"missing file {path}")

            # opens window to move images to correct folder
            if paths_to_check_move:
                for path in list(set(paths_to_check_move)):
                    show_folder_window(path)
                    paths_to_check_move = []

            # deletes needed files
            if delete_list:
                for path in list(set(delete_list)):
                    os.remove(path)
                    delete_list = []

    def check_matches(self):
        alt_sect = self._get_alt_path()
        for fl in self.paths["matches"]:
            if "_" + alt_sect not in fl[1] and "__" not in fl[1]:
                files = get_alternate_files(fl[1], alt_sect)
                origs = [self.file_copies[x] for x in files]
                delete_list, reorder_list = show_real_check(files, fl[0], self._get_desc(fl[0]), origs)
                delete_list_of_files(delete_list)
                rename_list_of_files(reorder_list)

    def view_matches(self):
        return self._out_data.loc[self._out_data["matches"] > 0]

    def view_partials(self):
        return self._out_data.loc[self._out_data["partials"] > 0]

    def view_missing(self):
        return self._out_data.loc[(self._out_data["partials"] == 0) and (self._out_data["matches"] == 0)]

    def reload_complete_df(self):
        df = self._out_data.drop(["matches", "partials"], axis=1)
        df["possible_filename"] = df.iloc[:, 0].apply(make_filename_valid)
        df["path"] = df.iloc[:, 0].apply(make_filename_valid)
        df["match"] = False

        found_files = self._get_files(self.out_path, [])
        for file_path in [x[0] for x in found_files]:
            _, file = os.path.split(file_path)
            file = make_filename_valid(file.rsplit('.', 1)[0].split("._alt")[0].split("_alt")[0])
            df.loc[df["possible_filename"] == file, "match"] = True

        return df
