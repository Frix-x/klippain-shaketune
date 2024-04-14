#!/usr/bin/env python3

############################################
###### INPUT SHAPER KLIPPAIN WORKFLOW ######
############################################
# Written by Frix_x#0161 #

#   This script is designed to be used with gcode_shell_commands directly from Klipper
#   Use the provided Shake&Tune macros instead!


import abc
import argparse
import glob
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

from git import GitCommandError, Repo

from analyze_axesmap import axesmap_calibration
from graph_belts import belts_calibration
from graph_shaper import shaper_calibration
from graph_vibrations import vibrations_profile
from locale_utils import print_with_c_locale


class Config:
    KLIPPER_FOLDER = os.path.expanduser('~/klipper')
    RESULTS_BASE_FOLDER = os.path.expanduser('~/printer_data/config/K-ShakeTune_results')
    RESULTS_SUBFOLDERS = {'belts': 'belts', 'shaper': 'inputshaper', 'vibrations': 'vibrations'}

    @staticmethod
    def get_results_folder(type):
        return os.path.join(Config.RESULTS_BASE_FOLDER, Config.RESULTS_SUBFOLDERS[type])

    @staticmethod
    def get_git_version():
        try:
            # Get the absolute path of the script, resolving any symlinks
            # Then get 1 times to parent dir to be at the git root folder
            script_path = Path(__file__).resolve()
            repo_path = script_path.parents[1]
            repo = Repo(repo_path)
            try:
                version = repo.git.describe('--tags')
            except GitCommandError:
                version = repo.head.commit.hexsha[:7]  # If no tag is found, use the simplified commit SHA instead
            return version
        except Exception:
            return None

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Shake&Tune graphs generation script')
        parser.add_argument(
            '-t',
            '--type',
            dest='type',
            choices=['belts', 'shaper', 'vibrations', 'axesmap', 'clean'],
            required=True,
            help='Type of output graph to produce',
        )
        parser.add_argument(
            '--accel',
            type=int,
            default=None,
            dest='accel_used',
            help='Accelerometion used for vibrations profile creation or axes map calibration',
        )
        parser.add_argument(
            '--chip_name',
            type=str,
            default='adxl345',
            dest='chip_name',
            help='Accelerometer chip name used for vibrations profile creation or axes map calibration',
        )
        parser.add_argument(
            '--max_smoothing',
            type=float,
            default=None,
            dest='max_smoothing',
            help='Maximum smoothing to allow for input shaper filter recommendations',
        )
        parser.add_argument(
            '--scv',
            '--square_corner_velocity',
            type=float,
            default=5.0,
            dest='scv',
            help='Square corner velocity used to compute max accel for input shapers filter recommendations',
        )
        parser.add_argument(
            '-m',
            '--kinematics',
            dest='kinematics',
            default='cartesian',
            choices=['cartesian', 'corexy'],
            help='Machine kinematics configuration used for the vibrations profile creation',
        )
        parser.add_argument(
            '-c',
            '--keep_csv',
            action='store_true',
            default=False,
            dest='keep_csv',
            help='Whether to keep the raw CSV files after processing in addition to the PNG graphs',
        )
        parser.add_argument(
            '-n',
            '--keep_results',
            type=int,
            default=3,
            dest='keep_results',
            help='Number of results to keep in the result folder after each run of the script',
        )
        parser.add_argument('--dpi', type=int, default=150, dest='dpi', help='DPI of the output PNG files')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + Config.get_git_version())

        return parser.parse_args()


class FileManager:
    @staticmethod
    def wait_file_ready(filepath):
        file_busy = True
        loop_count = 0
        while file_busy:
            for proc in os.listdir('/proc'):
                if proc.isdigit():
                    for fd in glob.glob(f'/proc/{proc}/fd/*'):
                        try:
                            if os.path.samefile(fd, filepath):
                                pass
                        except FileNotFoundError:  # Klipper has already released the CSV file
                            file_busy = False
                        except PermissionError:  # Unable to check for this particular process due to permissions
                            pass
            if loop_count > 60:
                # If Klipper is taking too long to release the file (60 * 1s = 1min), exit the script
                print_with_c_locale(f'Error: Klipper is taking too long to release {filepath}!')
                sys.exit(1)
            else:
                loop_count += 1
                time.sleep(1)
        return

    @staticmethod
    def ensure_folders_exist():
        for subfolder in Config.RESULTS_SUBFOLDERS.values():
            folder = os.path.join(Config.RESULTS_BASE_FOLDER, subfolder)
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def clean_old_files(type, keep_results=3, extension='.png'):
        folder = Config.get_results_folder(type)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]
        files.sort(key=os.path.getmtime, reverse=True)

        if 'belts' in folder:
            if len(files) <= keep_results + 1:
                return
            else:  # delete the older files
                for old_file in files[keep_results + 1 :]:
                    file_date = '_'.join(os.path.splitext(os.path.basename(old_file))[0].split('_')[1:3])
                    for suffix in ['A', 'B']:
                        csv_file = os.path.join(folder, f'belt_{file_date}_{suffix}.csv')
                        if os.path.exists(csv_file):
                            os.remove(csv_file)
                    os.remove(old_file)
        elif 'shaper' in folder:
            if len(files) <= 2 * keep_results + 1:
                return
            else:  # delete the older files
                for old_file in files[2 * keep_results + 1 :]:
                    csv_file = os.path.join(folder, os.path.splitext(os.path.basename(old_file))[0] + '.csv')
                    if os.path.exists(csv_file):
                        os.remove(csv_file)
                    os.remove(old_file)
        elif 'vibrations' in folder:
            if len(files) <= keep_results + 1:
                return
            else:  # delete the older files
                for old_file in files[keep_results + 1 :]:
                    os.remove(old_file)
                    tar_file = os.path.join(folder, os.path.splitext(os.path.basename(old_file))[0] + '.tar.gz')
                    if os.path.exists(tar_file):
                        os.remove(tar_file)


class GraphCreator(abc.ABC):
    def __init__(self, keep_csv, dpi):
        self._keep_csv = keep_csv
        self._dpi = dpi

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._version = Config.get_git_version()

    @abc.abstractmethod
    def create_graph(self):
        pass


class BeltsGraphCreator(GraphCreator):
    def __init__(self, keep_csv=False, dpi=150):
        super().__init__(keep_csv, dpi)

        self._type = 'belts'
        self._folder = Config.get_results_folder(self._type)

    def create_graph(self):
        globbed_files = glob.glob('/tmp/raw_data_axis*.csv')
        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to create the belt comparison graphs!')
        if len(globbed_files) < 2:
            raise FileNotFoundError('two CSV files are needed to create the belt comparison graphs!')

        lognames = []
        for filename in sorted(globbed_files, key=os.path.getmtime, reverse=True)[:2]:
            # Wait for the file handler to be released by Klipper
            FileManager.wait_file_ready(filename)

            # Cleanup of the filename and moving it in the result folder
            belt = os.path.basename(filename).split('_')[3].split('.')[0].upper()
            new_file = os.path.join(self._folder, f'{self._type}_{self._graph_date}_{belt}.csv')
            shutil.move(filename, new_file)
            lognames.append(new_file)

            # Check if the file is ready to be read
            os.sync()
            FileManager.wait_file_ready(new_file)

        fig = belts_calibration(lognames, Config.KLIPPER_FOLDER, self._version)
        png_filename = os.path.join(self._folder, f'{self._type}_{self._graph_date}.png')
        fig.savefig(png_filename, dpi=self._dpi)

        # Remove the CSV files if the user don't want to keep them
        if not self._keep_csv:
            for csv in lognames:
                if os.path.exists(csv):
                    os.remove(csv)


class ShaperGraphCreator(GraphCreator):
    def __init__(self, max_smoothing=None, scv=5.0, keep_csv=False, dpi=150):
        super().__init__(keep_csv, dpi)

        self._max_smoothing = max_smoothing
        self._scv = scv

        self._type = 'shaper'
        self._folder = Config.get_results_folder(self._type)

    def create_graph(self):
        globbed_files = glob.glob('/tmp/raw_data*.csv')
        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to create the input shaper graphs!')

        # Find the CSV files with the latest timestamp and wait for it to be released by Klipper
        filename = sorted(globbed_files, key=os.path.getmtime, reverse=True)[0]
        FileManager.wait_file_ready(filename)

        # Cleanup of the filename and moving it in the result folder
        axis = os.path.basename(filename).split('_')[3].split('.')[0].upper()
        new_file = os.path.join(self._folder, f'{self._type}_{self._graph_date}_{axis}.csv')
        shutil.move(filename, new_file)

        # Check if the file is ready to be read
        os.sync()
        FileManager.wait_file_ready(new_file)

        fig = shaper_calibration(
            [new_file],
            Config.KLIPPER_FOLDER,
            max_smoothing=self._max_smoothing,
            scv=self._scv,
            st_version=self._version,
        )
        png_filename = os.path.join(self._folder, f'{self._type}_{self._graph_date}_{axis}.png')
        fig.savefig(png_filename, dpi=self._dpi)

        # Remove the CSV files if the user don't want to keep them
        if not self._keep_csv:
            if os.path.exists(new_file):
                os.remove(new_file)

        return axis


class VibrationsGraphCreator(GraphCreator):
    def __init__(self, kinematics, accel, chip_name, keep_csv=False, dpi=150):
        super().__init__(keep_csv, dpi)

        self._kinematics = kinematics
        self._accel = accel
        self._chip_name = chip_name

        self._type = 'vibrations'
        self._folder = Config.get_results_folder(self._type)

    def create_graph(self):
        globbed_files = glob.glob(f'/tmp/{self._chip_name}-*.csv')
        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to create the vibrations graphs!')
        if len(globbed_files) < 3:
            raise FileNotFoundError('at least 3 CSV files are needed to create the vibrations graphs!')

        lognames = []
        for filename in globbed_files:
            # Wait for the file handler to be released by Klipper
            FileManager.wait_file_ready(filename)

            # Cleanup of the filename and moving it in the result folder
            cleanfilename = os.path.basename(filename).replace(self._chip_name, f'{self._type}_{self._graph_date}')
            new_file = os.path.join(self._folder, cleanfilename)
            shutil.move(filename, new_file)
            lognames.append(new_file)

        # Sync filesystem to avoid problems as there is a lot of file copied
        os.sync()
        time.sleep(5)

        fig = vibrations_profile(lognames, Config.KLIPPER_FOLDER, self._kinematics, self._accel, self._version)
        png_filename = os.path.join(self._folder, f'{self._type}_{self._graph_date}.png')
        fig.savefig(png_filename, dpi=self._dpi)

        # Archive all the csv files in a tarball in case the user want to keep them
        if self._keep_csv:
            with tarfile.open(os.path.join(self._folder, f'{self._type}_{self._graph_date}.tar.gz'), 'w:gz') as tar:
                for csv_file in lognames:
                    tar.add(csv_file, arcname=os.path.basename(csv_file), recursive=False)

        # Remove the remaining CSV files not needed anymore (tarball is safe if it was created)
        for csv in lognames:
            if os.path.exists(csv):
                os.remove(csv)


class AxesMapFinder:
    def __init__(self, accel, chip_name):
        self._accel = accel
        self._chip_name = chip_name

        self._graph_date = datetime.now().strftime('%Y%m%d_%H%M%S')

        self._type = 'axesmap'
        self._folder = Config.RESULTS_BASE_FOLDER

    def find_axesmap(self):
        globbed_files = glob.glob(f'/tmp/{self._chip_name}-*.csv')
        if not globbed_files:
            raise FileNotFoundError('no CSV files found in the /tmp folder to find the axes map!')

        # Find the CSV files with the latest timestamp and wait for it to be released by Klipper
        logname = sorted(globbed_files, key=os.path.getmtime, reverse=True)[0]
        FileManager.wait_file_ready(logname)

        results = axesmap_calibration([logname], self._accel)
        result_filename = os.path.join(self._folder, f'{self._type}_{self._graph_date}.txt')
        with open(result_filename, 'w') as f:
            f.write(results)


def main():
    print_with_c_locale(f'Shake&Tune version: {Config.get_git_version()}')

    options = Config.parse_arguments()
    FileManager.ensure_folders_exist()

    graph_creator = None
    if options.type == 'belts':
        graph_creator = BeltsGraphCreator(options.keep_csv, options.dpi)
    elif options.type == 'shaper':
        graph_creator = ShaperGraphCreator(options.max_smoothing, options.scv, options.keep_csv, options.dpi)
    elif options.type == 'vibrations':
        graph_creator = VibrationsGraphCreator(
            options.kinematics, options.accel_used, options.chip_name, options.keep_csv, options.dpi
        )
    elif options.type == 'axesmap':
        graph_creator = AxesMapFinder(options.accel_used, options.chip_name)
    elif options.type == 'clean':
        print_with_c_locale(f'Cleaning output folder to keep only the last {options.keep_results} results...')
        FileManager.clean_old_files(options.type, options.keep_results)
        return

    if graph_creator:
        graph_creator.create_graph()
        print_with_c_locale(f'{options.type} graphs created successfully!')

    if options.keep_csv is False and options.type != 'clean':
        print_with_c_locale('Deleting raw CSV files... If you want to keep them, use the --keep_csv option!')


if __name__ == '__main__':
    main()
