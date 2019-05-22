import subprocess
import os
import zipfile
from pathlib import Path

KAGGLE = 'kaggle'
COMPETITIONS = 'competitions'
DOWNLOAD = 'download'
SUBMIT = 'submit'


def downaload(comp_name, path=None):

    args = [
        KAGGLE,
        COMPETITIONS,
        DOWNLOAD,
        '-c',
        comp_name,
        '-p',
        path
    ]

    subprocess.run(args)

    _extract_all_zip(path)


def _extract_all_zip(directory_path):
    path = Path(directory_path)

    for filename in os.listdir(path):
        if filename.endswith('.zip'):
            _extract_zip(path, filename)


def _extract_zip(path, filename):
    print('Unzip file %s' % filename)

    zip = zipfile.ZipFile(path / filename)
    zip.extractall(path)
    zip.close()

    os.remove(path / filename)


def submit(comp_name, filename, message):

    args = [
        KAGGLE,
        COMPETITIONS,
        SUBMIT,
        comp_name,
        '-f',
        filename,
        '-m',
        message
    ]

    subprocess.run(args)
