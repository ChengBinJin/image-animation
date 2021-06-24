import os


def all_files_under(path, extension=None, append_path=True, sort=True):
    if not isinstance(extension, list):
        extension = [extension]

    filenames = []
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            for fname in os.listdir(path):
                if os.path.splitext(fname)[1] in extension:
                    filenames.append(os.path.join(path, fname))
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            for fname in os.listdir(path):
                if os.path.splitext(fname)[1] in extension:
                    filenames.append(os.path.basename(fname))

    if sort:
        filenames = sorted(filenames)

    return filenames


def get_name_and_ext(path):
    base_name = os.path.basename(path)
    name, ext = os.path.splitext(base_name)

    while '.' in name:
        name, _ = os.path.splitext(name)

    return name, ext


def get_name(path):
    return get_name_and_ext(path)[0]


def get_ext(path):
    return get_name_and_ext(path)[1]
