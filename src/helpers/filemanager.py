#!/usr/bin/env python3

# Common file management functions for the Shake&Tune package
# Written by Frix_x#0161 #

import os
import time
from pathlib import Path


def wait_file_ready(filepath: Path, timeout: int = 60) -> None:
    file_busy = True
    loop_count = 0

    while file_busy:
        if loop_count >= timeout:
            raise TimeoutError(f'Klipper is taking too long to release the CSV file ({filepath})!')

        # Try to open the file in write-only mode to check if it is in use
        # If we successfully open and close the file, it is not in use
        try:
            fd = os.open(filepath, os.O_WRONLY)
            os.close(fd)
            file_busy = False
        except OSError:
            # If OSError is caught, it indicates the file is still being used
            pass
        except Exception:
            # If another exception is raised, it's not a problem, we just loop again
            pass

        loop_count += 1
        time.sleep(1)


def ensure_folders_exist(folders: list[Path]) -> None:
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
