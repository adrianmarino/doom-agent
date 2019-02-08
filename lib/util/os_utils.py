import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_file_path(path, filename, ext=''):
    return os.path.join(create_path(path), f'{filename}{"." if ext else ""}{ext}')


def last_created_file_from(path):
    try:
        latest_file = max(path, key=os.path.getctime)
        return latest_file
    except:
        return None
