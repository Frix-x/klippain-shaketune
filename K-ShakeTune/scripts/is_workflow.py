#!/usr/bin/env python3

############################################
###### INPUT SHAPER KLIPPAIN WORKFLOW ######
############################################
# Written by Frix_x#0161 #

#   This script is designed to be used with gcode_shell_commands directly from Klipper
#   Use the provided Shake&Tune macros instead!


import optparse
import os
import time
import glob
import sys
import shutil
import tarfile
from datetime import datetime

#################################################################################################################
RESULTS_FOLDER = os.path.expanduser('~/printer_data/config/K-ShakeTune_results')
KLIPPER_FOLDER = os.path.expanduser('~/klipper')
#################################################################################################################

from graph_belts import belts_calibration
from graph_shaper import shaper_calibration
from graph_vibrations import vibrations_calibration
from analyze_axesmap import axesmap_calibration

RESULTS_SUBFOLDERS = ['belts', 'inputshaper', 'vibrations']


def is_file_open(filepath):
    for proc in os.listdir('/proc'):
        if proc.isdigit():
            for fd in glob.glob(f'/proc/{proc}/fd/*'):
                try:
                    if os.path.samefile(fd, filepath):
                        return True
                except FileNotFoundError:
                    # Klipper has already released the CSV file
                    pass
                except PermissionError:
                    # Unable to check for this particular process due to permissions
                    pass
    return False


def create_belts_graph(keep_csv):
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    lognames = []

    globbed_files = glob.glob('/tmp/raw_data_axis*.csv')
    if not globbed_files:
        print("No CSV files found in the /tmp folder to create the belt graphs!")
        sys.exit(1)
    if len(globbed_files) < 2:
        print("Not enough CSV files found in the /tmp folder. Two files are required for the belt graphs!")
        sys.exit(1)

    sorted_files = sorted(globbed_files, key=os.path.getmtime, reverse=True)

    for filename in sorted_files[:2]:
        # Wait for the file handler to be released by Klipper
        while is_file_open(filename):
            time.sleep(2)
        
        # Extract the tested belt from the filename and rename/move the CSV file to the result folder
        belt = os.path.basename(filename).split('_')[3].split('.')[0].upper()
        new_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[0], f'belt_{current_date}_{belt}.csv')
        shutil.move(filename, new_file)
        os.sync() # Sync filesystem to avoid problems

        # Save the file path for later
        lognames.append(new_file)

        # Wait for the file handler to be released by the move command
        while is_file_open(new_file):
            time.sleep(2)
    
    # Generate the belts graph and its name
    fig = belts_calibration(lognames, KLIPPER_FOLDER)
    png_filename = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[0], f'belts_{current_date}.png')
    fig.savefig(png_filename, dpi=150)
    
    # Remove the CSV files if the user don't want to keep them
    if not keep_csv:
        for csv in lognames:
            if os.path.exists(csv):
                os.remove(csv)

    return


def create_shaper_graph(keep_csv, max_smoothing, scv):
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get all the files and sort them based on last modified time to select the most recent one
    globbed_files = glob.glob('/tmp/raw_data*.csv')
    if not globbed_files:
        print("No CSV files found in the /tmp folder to create the input shaper graphs!")
        sys.exit(1)

    sorted_files = sorted(globbed_files, key=os.path.getmtime, reverse=True)
    filename = sorted_files[0]

    # Wait for the file handler to be released by Klipper
    while is_file_open(filename):
        time.sleep(2)
    
    # Extract the tested axis from the filename and rename/move the CSV file to the result folder
    axis = os.path.basename(filename).split('_')[3].split('.')[0].upper()
    new_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[1], f'resonances_{current_date}_{axis}.csv')
    shutil.move(filename, new_file)
    os.sync() # Sync filesystem to avoid problems

    # Wait for the file handler to be released by the move command
    while is_file_open(new_file):
        time.sleep(2)
    
    # Generate the shaper graph and its name
    fig = shaper_calibration([new_file], KLIPPER_FOLDER, max_smoothing=max_smoothing, scv=scv)
    png_filename = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[1], f'resonances_{current_date}_{axis}.png')
    fig.savefig(png_filename, dpi=150)

    # Remove the CSV file if the user don't want to keep it
    if not keep_csv:
        if os.path.exists(new_file):
            os.remove(new_file)
    
    return axis


def create_vibrations_graph(axis_name, accel, chip_name, keep_csv):
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    lognames = []

    globbed_files = glob.glob(f'/tmp/{chip_name}-*.csv')
    if not globbed_files:
        print("No CSV files found in the /tmp folder to create the vibration graphs!")
        sys.exit(1)
    if len(globbed_files) < 3:
        print("Not enough CSV files found in the /tmp folder. At least 3 files are required for the vibration graphs!")
        sys.exit(1)

    for filename in globbed_files:
        # Wait for the file handler to be released by Klipper
        while is_file_open(filename):
            time.sleep(2)

        # Cleanup of the filename and moving it in the result folder
        cleanfilename = os.path.basename(filename).replace(chip_name, f'vibr_{current_date}')
        new_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[2], cleanfilename)
        shutil.move(filename, new_file)

        # Save the file path for later
        lognames.append(new_file)

    # Sync filesystem to avoid problems as there is a lot of file copied
    os.sync()
    time.sleep(5)

    # Generate the vibration graph and its name
    fig = vibrations_calibration(lognames, KLIPPER_FOLDER, axis_name, accel)
    png_filename = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[2], f'vibrations_{current_date}_{axis_name}.png')
    fig.savefig(png_filename, dpi=150)
    
    # Archive all the csv files in a tarball in case the user want to keep them
    if keep_csv:
        with tarfile.open(os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[2], f'vibrations_{current_date}_{axis_name}.tar.gz'), 'w:gz') as tar:
            for csv_file in lognames:
                tar.add(csv_file, recursive=False)

    # Remove the remaining CSV files not needed anymore (tarball is safe if it was created)
    for csv_file in lognames:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    return


def find_axesmap(accel, chip_name):
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = os.path.join(RESULTS_FOLDER, f'axes_map_{current_date}.txt')
    lognames = []

    globbed_files = glob.glob(f'/tmp/{chip_name}-*.csv')
    if not globbed_files:
        print("No CSV files found in the /tmp folder to analyze and find the axes_map!")
        sys.exit(1)

    sorted_files = sorted(globbed_files, key=os.path.getmtime, reverse=True)
    filename = sorted_files[0]

    # Wait for the file handler to be released by Klipper
    while is_file_open(filename):
        time.sleep(2)

    # Analyze the CSV to find the axes_map parameter
    lognames.append(filename)
    results = axesmap_calibration(lognames, accel)

    with open(result_filename, 'w') as f:
        f.write(results)

    return


# Utility function to get old files based on their modification time
def get_old_files(folder, extension, limit):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[limit:]

def clean_files(keep_results):
    # Define limits based on STORE_RESULTS
    keep1 = keep_results + 1
    keep2 = 2 * keep_results + 1

    # Find old files in each directory
    old_belts_files = get_old_files(os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[0]), '.png', keep1)
    old_inputshaper_files = get_old_files(os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[1]), '.png', keep2)
    old_vibrations_files = get_old_files(os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[2]), '.png', keep1)

    # Remove the old belt files
    for old_file in old_belts_files:
        file_date = "_".join(os.path.splitext(os.path.basename(old_file))[0].split('_')[1:3])
        for suffix in ['A', 'B']:
            csv_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[0], f'belt_{file_date}_{suffix}.csv')
            if os.path.exists(csv_file):
                os.remove(csv_file)
        os.remove(old_file)
    
    # Remove the old shaper files
    for old_file in old_inputshaper_files:
        csv_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[1], os.path.splitext(os.path.basename(old_file))[0] + ".csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)
        os.remove(old_file)

    # Remove the old vibrations files
    for old_file in old_vibrations_files:
        os.remove(old_file)
        tar_file = os.path.join(RESULTS_FOLDER, RESULTS_SUBFOLDERS[2], os.path.splitext(os.path.basename(old_file))[0] + ".tar.gz")
        if os.path.exists(tar_file):
            os.remove(tar_file)


def main():
    # Parse command-line arguments
    usage = "%prog [options] <logs>"
    opts = optparse.OptionParser(usage)
    opts.add_option("-t", "--type", type="string", dest="type",
                    default=None, help="type of output graph to produce")
    opts.add_option("--accel", type="int", default=None, dest="accel_used",
                    help="acceleration used during the vibration macro or axesmap macro")
    opts.add_option("--axis_name", type="string", default=None, dest="axis_name",
                    help="axis tested during the vibration macro")
    opts.add_option("--chip_name", type="string", default="adxl345", dest="chip_name",
                    help="accelerometer chip name in klipper used during the vibration macro or the axesmap macro")
    opts.add_option("-n", "--keep_results", type="int", default=3, dest="keep_results",
                    help="number of results to keep in the result folder after each run of the script")
    opts.add_option("-c", "--keep_csv", action="store_true", default=False, dest="keep_csv",
                    help="weither or not to keep the CSV files alongside the PNG graphs image results")
    opts.add_option("--scv", "--square_corner_velocity", type="float", dest="scv", default=5.,
                    help="square corner velocity used to compute max accel for axis shapers graphs")
    opts.add_option("--max_smoothing", type="float", dest="max_smoothing", default=None,
                    help="maximum shaper smoothing to allow")
    options, args = opts.parse_args()
    
    if options.type is None:
        opts.error("You must specify the type of output graph you want to produce (option -t)")
    elif options.type.lower() is None or options.type.lower() not in ['belts', 'shaper', 'vibrations', 'axesmap', 'clean']:
        opts.error("Type of output graph need to be in the list of 'belts', 'shaper', 'vibrations', 'axesmap' or 'clean'")
    else:
        graph_mode = options.type

    # Check if results folders are there or create them before doing anything else
    for result_subfolder in RESULTS_SUBFOLDERS:
        folder = os.path.join(RESULTS_FOLDER, result_subfolder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    if graph_mode.lower() == 'belts':
        create_belts_graph(keep_csv=options.keep_csv)
        print(f"Belt graph created. You will find the results in {RESULTS_FOLDER}/{RESULTS_SUBFOLDERS[0]}")
    elif graph_mode.lower() == 'shaper':
        axis = create_shaper_graph(keep_csv=options.keep_csv, max_smoothing=options.max_smoothing, scv=options.scv)
        print(f"{axis} input shaper graph created. You will find the results in {RESULTS_FOLDER}/{RESULTS_SUBFOLDERS[1]}")
    elif graph_mode.lower() == 'vibrations':
        create_vibrations_graph(axis_name=options.axis_name, accel=options.accel_used, chip_name=options.chip_name, keep_csv=options.keep_csv)
        print(f"{options.axis_name} vibration graph created. You will find the results in {RESULTS_FOLDER}/{RESULTS_SUBFOLDERS[2]}")
    elif graph_mode.lower() == 'axesmap':
        print(f"WARNING: AXES_MAP_CALIBRATION is currently very experimental and may produce incorrect results... Please validate the output!")
        find_axesmap(accel=options.accel_used, chip_name=options.chip_name)
    elif graph_mode.lower() == 'clean':
        print(f"Cleaning output folder to keep only the last {options.keep_results} results...")
        clean_files(keep_results=options.keep_results)


if __name__ == '__main__':
    main()
