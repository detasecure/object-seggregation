import os
import time
import datetime
from pathlib import Path
import imghdr
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO


UPLOAD_FOLDER = 'static'
UPLOAD_EXTENSIONS = ['.jpg', '.png', 'jpeg']


def is_image_content(file_content):
    try:
        Image.open(BytesIO(file_content))
        return True
    except IOError:
        return False


def remove_files():
    for f in [x for x in Path(UPLOAD_FOLDER).iterdir() if x.is_file() and not x.name.startswith('.')]:
        creation_date = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        now = datetime.datetime.now()
        diff = now - creation_date
        if diff.seconds > 60:
            print(f'Removing file {f.name}')
            os.remove(f)

def save_file(filename, file):

    filename = secure_filename(filename)

    if filename == '':
        return (False, 'Empty filename')
    elif not filename.lower().endswith(tuple(UPLOAD_EXTENSIONS)):
        return (False, f"The file extension should be {UPLOAD_EXTENSIONS}")

    image_file_content = file.read()

    if not is_image_content(image_file_content):
        return (False, f"Not an Image file")

    # remove_files()

    timestamp = time.time()
    fullname = os.path.join(UPLOAD_FOLDER, filename + f'_{timestamp}')
    with open(fullname, "wb") as buffer:
        buffer.write(image_file_content)

    return (True, fullname, image_file_content)

