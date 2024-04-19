#!/usr/bin/env python3

# Common file management functions for the Shake&Tune package
# Written by Frix_x#0161 #

import time
from pathlib import Path

from is_workflow import Config


def wait_file_ready(filepath: Path) -> None:
    file_busy = True
    loop_count = 0
    proc_path = Path('/proc')
    while file_busy:
        if loop_count > 60:
            # If Klipper is taking too long to release the file (60 * 1s = 1min), raise an error
            raise TimeoutError(f'Klipper is taking too long to release {filepath}!')

        for proc in proc_path.iterdir():
            if proc.name.isdigit():
                fd_path = proc / 'fd'
                if fd_path.exists():
                    for fd in fd_path.iterdir():
                        try:
                            # Using resolve to ensure symbolic links are followed
                            if fd.resolve(strict=False) == filepath:
                                pass  # File is still being used by Klipper
                        except FileNotFoundError:  # Klipper has already released the CSV file
                            file_busy = False
                            break
                        except PermissionError:  # Unable to check for this particular process due to permissions
                            pass

        loop_count += 1
        time.sleep(1)


def ensure_folders_exist() -> None:
    for subfolder in Config.RESULTS_SUBFOLDERS.values():
        folder = Config.RESULTS_BASE_FOLDER / subfolder
        Path(folder).mkdir(parents=True, exist_ok=True)
