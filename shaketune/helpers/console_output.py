# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: console_output.py
# Description: Defines the ConsoleOutput class for printing output to stdout or an alternative
#              callback function, such as the Klipper console.


import io
from typing import Callable, Optional


class ConsoleOutput:
    """
    Print output to stdout or to an alternative like the Klipper console through a callback
    """

    _output_func: Optional[Callable[[str], None]] = None

    @classmethod
    def register_output_callback(cls, output_func: Optional[Callable[[str], None]]):
        cls._output_func = output_func

    @classmethod
    def print(cls, *args, **kwargs):
        if not cls._output_func:
            print(*args, **kwargs)
            return

        with io.StringIO() as mem_output:
            print(*args, file=mem_output, **kwargs)
            cls._output_func(mem_output.getvalue())
