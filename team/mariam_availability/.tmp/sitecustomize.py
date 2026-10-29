import os, shutil, tempfile

def _workspace_safe_resetperms(path):
    try:
        os.chmod(path, 0o700)
    except PermissionError:
        pass

shutil._resetperms = _workspace_safe_resetperms
tempfile._resetperms = _workspace_safe_resetperms
