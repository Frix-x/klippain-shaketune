#!/usr/bin/env python3

# Special utility functions to manage locale settings and printing
# Written by Frix_x#0161 #


import io
import locale
from typing import Callable, Optional

KLIPPER_CONSOLE_OUTPUT_FUNC: Optional[Callable[[str], None]] = None


def set_shaketune_output_func(func: Callable[[str], None]):
    global KLIPPER_CONSOLE_OUTPUT_FUNC
    KLIPPER_CONSOLE_OUTPUT_FUNC = func


# Set the best locale for time and date formating (generation of the titles)
def set_locale():
    try:
        current_locale = locale.getlocale(locale.LC_TIME)
        if current_locale is None or current_locale[0] is None:
            locale.setlocale(locale.LC_TIME, 'C')
    except locale.Error:
        locale.setlocale(locale.LC_TIME, 'C')


# Print function to avoid problem in Klipper console (that doesn't support special characters) due to locale settings
def print_with_c_locale(*args, **kwargs):
    try:
        original_locale = locale.getlocale()
        locale.setlocale(locale.LC_ALL, 'C')
    except locale.Error as e:
        print(
            'Warning: Failed to set a basic locale. Special characters may not display correctly in Klipper console:', e
        )
    finally:
        if not KLIPPER_CONSOLE_OUTPUT_FUNC:
            print(*args, **kwargs)  # Proceed with printing regardless of locale setting success
        else:
            with io.StringIO() as mem_output:
                print(*args, file=mem_output, **kwargs)
                KLIPPER_CONSOLE_OUTPUT_FUNC(mem_output.getvalue())
        try:
            locale.setlocale(locale.LC_ALL, original_locale)
        except locale.Error as e:
            print('Warning: Failed to restore the original locale setting:', e)
