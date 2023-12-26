#!/usr/bin/env python3

# Special utility functions to manage locale settings and printing
# Written by Frix_x#0161 #


import locale

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
        print("Warning: Failed to set a basic locale. Special characters may not display correctly in Klipper console:", e)
    finally:
        print(*args, **kwargs) # Proceed with printing regardless of locale setting success
        try:
            locale.setlocale(locale.LC_ALL, original_locale)
        except locale.Error as e:
            print("Warning: Failed to restore the original locale setting:", e)
