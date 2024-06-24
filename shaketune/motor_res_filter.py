# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: motor_res_filter.py
# Description: This script defines the MotorResonanceFilter class that applies and removes motor resonance filters
#              into the input shaper initial Klipper object. This is done by convolving a motor resonance targeted
#              input shaper filter with the current configured axis input shapers.

from importlib import import_module

from .helpers.console_output import ConsoleOutput

shaper_defs = import_module('.shaper_defs', 'extras')


class MotorResonanceFilter:
    def __init__(self, printer, freq_x: float, freq_y: float, damping_x: float, damping_y: float, in_danger: bool):
        self._printer = printer
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.damping_x = damping_x
        self.damping_y = damping_y
        self._in_danger = in_danger

        self._original_shapers = {}

    # Convolve two Klipper shapers into a new composite shaper
    @staticmethod
    def convolve_shapers(L, R):
        As = [a * b for a in L[0] for b in R[0]]
        Ts = [a + b for a in L[1] for b in R[1]]
        C = sorted(list(zip(Ts, As)))
        return ([a for _, a in C], [t for t, _ in C])

    def apply_filters(self) -> None:
        input_shaper = self._printer.lookup_object('input_shaper', None)
        shapers = input_shaper.get_shapers()
        for shaper in shapers:
            axis = shaper.axis
            shaper_type = shaper.params.get_status()['shaper_type']

            # Ignore the motor resonance filters for smoothers from DangerKlipper
            if shaper_type.startswith('smooth_'):
                ConsoleOutput.print(
                    (
                        f'Warning: {shaper_type} type shaper on {axis} axis is a smoother from DangerKlipper '
                        'Bleeding-Edge that already filters the motor resonance frequency range. Shake&Tune '
                        'motor resonance filters will be ignored for this axis...'
                    )
                )
                continue

            # Ignore the motor resonance filters for custom shapers as users can set their own A&T values
            if shaper_type == 'custom':
                ConsoleOutput.print(
                    (
                        f'Warning: custom type shaper on {axis} axis is a manually crafted filter. So you have '
                        'already set custom A&T values for this axis and you should be able to convolve the motor '
                        'resonance frequency range to this custom shaper. Shake&Tune motor resonance filters will '
                        'be ignored for this axis...'
                    )
                )
                continue

            # At the moment, when running stock Klipper, only ZV type shapers are supported to get combined with
            # the motor resonance filters. This is due to the size of the pulse train that is too small and is not
            # allowing the convolved shapers to be applied. This unless this PR is merged: https://github.com/Klipper3d/klipper/pull/6460
            if not self._in_danger and shaper_type != 'zv':
                ConsoleOutput.print(
                    (
                        f'Error: the {axis} axis is not a ZV type shaper. Shake&Tune motor resonance filters '
                        'will be ignored for this axis... Thi is due to the size of the pulse train being too '
                        'small and not allowing the convolved shapers to be applied... unless this PR is '
                        'merged: https://github.com/Klipper3d/klipper/pull/6460'
                    )
                )
                continue

            # Get the current shaper parameters and store them for later restoration
            _, A, T = shaper.get_shaper()
            self._original_shapers[axis] = (A, T)

            # Creating the new combined shapers that contains the motor resonance filters
            if axis in {'x', 'y'}:
                if self._in_danger:
                    # In DangerKlipper, the pulse train is large enough to allow the
                    # convolution of any shapers in order to craft the new combined shapers
                    new_A, new_T = MotorResonanceFilter.convolve_shapers(
                        (A, T),
                        shaper_defs.get_mzv_shaper(self.freq_x, self.damping_x),
                    )
                else:
                    # In stock Klipper, the pulse train is too small for most shapers
                    # to be convolved. So we need to use the ZV shaper instead for the
                    # motor resonance filters... even if it's not the best for this purpose
                    new_A, new_T = MotorResonanceFilter.convolve_shapers(
                        (A, T),
                        shaper_defs.get_zv_shaper(self.freq_x, self.damping_x),
                    )

                shaper.A = new_A
                shaper.T = new_T
                shaper.n = len(new_A)

        # Update the running input shaper filter with the new parameters
        input_shaper._update_input_shaping()

    def remove_filters(self) -> None:
        input_shaper = self._printer.lookup_object('input_shaper', None)
        shapers = input_shaper.get_shapers()
        for shaper in shapers:
            axis = shaper.axis
            if axis in self._original_shapers:
                A, T = self._original_shapers[axis]
                shaper.A = A
                shaper.T = T
                shaper.n = len(A)

        # Update the running input shaper filter with the restored initial parameters
        # to keep only standard axis input shapers activated
        input_shaper._update_input_shaping()
