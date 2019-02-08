import glob
import os


def last_created_file_from(path):
    try:
        list_of_files = glob.glob(path)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    except:
        return None
