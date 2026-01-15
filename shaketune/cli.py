import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

from .graph_creators.graph_creator_factory import GraphCreatorFactory
from .helpers.accelerometer import MeasurementsManager
from .shaketune_config import ShakeTuneConfig


def add_common_arguments(parser):
    """Helper function to add common arguments to all subparsers."""
    parser.add_argument('-o', '--output', required=True, help='Output filename')
    parser.add_argument('files', nargs='+', help='Input data files (.csv or .stdata)')
    parser.add_argument('--max_freq', type=float, help='Maximum frequency to graph')
    parser.add_argument('--dpi', type=int, help='DPI value to use for the graph')


def configure_graph_creator(graph_type, args, dummy_config):
    """Helper function to get and configure a graph creator based on graph type and args."""
    graph_creator = GraphCreatorFactory.create_graph_creator(graph_type, dummy_config)
    config_kwargs = {}

    # Dynamically configure the graph creator based on graph type
    if graph_type == 'axes map':
        config_kwargs |= {'accel': args.accel, 'current_axes_map': args.axes_map}
    elif graph_type == 'static frequency':
        config_kwargs |= {'accel_per_hz': args.accel_per_hz, 'freq': args.frequency, 'duration': args.duration}
    elif graph_type == 'belts comparison':
        config_kwargs |= {
            'kinematics': args.kinematics,
            'test_params': (args.mode, None, None, args.accel_per_hz, None, args.sweeping_accel, args.sweeping_period),
            'max_scale': args.max_scale,
        }
    elif graph_type == 'input shaper':
        config_kwargs |= {
            'scv': args.scv,
            'max_smoothing': args.max_smoothing,
            'test_params': (args.mode, None, None, args.accel_per_hz, None, args.sweeping_accel, args.sweeping_period),
            'max_scale': args.max_scale,
        }
    elif graph_type == 'vibrations profile':
        config_kwargs |= {'kinematics': args.kinematics, 'accel': args.accel}

    graph_creator.configure(**config_kwargs)
    return graph_creator


def load_klipper_module(args):
    """Helper function to load the shaper_calibrate module from the specified Klipper folder."""
    if hasattr(args, 'klipper_dir') and args.klipper_dir:
        kdir = os.path.expanduser(args.klipper_dir)
        sys.path.append(os.path.join(kdir, 'klippy'))
        sys.modules['shaper_calibrate'] = import_module('.shaper_calibrate', 'extras')
        sys.modules['shaper_defs'] = import_module('.shaper_defs', 'extras')


def main():
    parser = argparse.ArgumentParser(description='Shake&Tune command line interface')
    subparsers = parser.add_subparsers(dest='graph_type', help='Type of graph to create')

    # Static frequency graph parser
    static_freq_parser = subparsers.add_parser('static_freq', help='Create static frequency graph')
    add_common_arguments(static_freq_parser)
    static_freq_parser.add_argument('--accel_per_hz', type=float, help='Accel per Hz used during the measurement')
    static_freq_parser.add_argument('--frequency', type=float, help='Maintained frequency of the measurement')
    static_freq_parser.add_argument('--duration', type=float, help='Duration of the measurement')

    # Axes map detection graph parser
    axes_map_parser = subparsers.add_parser('axes_map', help='Create axes map detection graph')
    add_common_arguments(axes_map_parser)
    axes_map_parser.add_argument('--accel', required=True, type=float, help='Accel value used for the measurement')
    axes_map_parser.add_argument(
        '--axes_map',
        type=str,
        help='Existing axes_map config to invert. Use = syntax when value starts with "-" (e.g., --axes_map="-y,x,z")',
    )

    # Belts graph parser
    belts_parser = subparsers.add_parser('belts', help='Create belts comparison graph')
    add_common_arguments(belts_parser)
    belts_parser.add_argument('-k', '--klipper_dir', default='~/klipper', help='Main klipper directory')
    belts_parser.add_argument('--kinematics', help='Machine kinematics configuration')
    belts_parser.add_argument('--mode', type=str, help='Mode of the test used during the measurement')
    belts_parser.add_argument('--accel_per_hz', type=float, help='Accel per Hz used during the measurement')
    belts_parser.add_argument(
        '--sweeping_accel', type=float, help='Accel used during the sweeping test (if sweeping was used)'
    )
    belts_parser.add_argument(
        '--sweeping_period', type=float, help='Sweeping period used during the sweeping test (if sweeping was used)'
    )
    belts_parser.add_argument(
        '--max_scale', type=lambda x: int(float(x)), help='Maximum energy value to scale the belts graph'
    )

    # Input Shaper graph parser
    shaper_parser = subparsers.add_parser('input_shaper', help='Create input shaper graph')
    add_common_arguments(shaper_parser)
    shaper_parser.add_argument('-k', '--klipper_dir', default='~/klipper', help='Main klipper directory')
    shaper_parser.add_argument('--scv', type=float, default=5.0, help='Square corner velocity')
    shaper_parser.add_argument('--max_smoothing', type=float, help='Maximum shaper smoothing to allow')
    shaper_parser.add_argument('--mode', type=str, help='Mode of the test used during the measurement')
    shaper_parser.add_argument('--accel_per_hz', type=float, help='Accel per Hz used during the measurement')
    shaper_parser.add_argument(
        '--sweeping_accel', type=float, help='Accel used during the sweeping test (if sweeping was used)'
    )
    shaper_parser.add_argument(
        '--sweeping_period', type=float, help='Sweeping period used during the sweeping test (if sweeping was used)'
    )
    shaper_parser.add_argument(
        '--max_scale', type=lambda x: int(float(x)), help='Maximum energy value to scale the input shaper graph'
    )

    # Vibrations graph parser
    vibrations_parser = subparsers.add_parser('vibrations', help='Create vibrations profile graph')
    add_common_arguments(vibrations_parser)
    vibrations_parser.add_argument('-k', '--klipper_dir', default='~/klipper', help='Main klipper directory')
    vibrations_parser.add_argument('--kinematics', required=True, default='cartesian', help='Used kinematics')
    vibrations_parser.add_argument('--accel', type=int, help='Accel value to be printed on the graph')

    args = parser.parse_args()

    if args.graph_type is None:
        parser.print_help()
        exit(1)

    graph_type_map = {
        'static_freq': 'static frequency',
        'axes_map': 'axes map',
        'belts': 'belts comparison',
        'input_shaper': 'input shaper',
        'vibrations': 'vibrations profile',
    }
    graph_type = graph_type_map[args.graph_type]

    # Load configuration
    dummy_config = ShakeTuneConfig()
    if args.dpi is not None:
        dummy_config.dpi = args.dpi
    if args.max_freq is not None:
        if graph_type == 'vibrations profile':
            dummy_config.max_freq_vibrations = args.max_freq
        else:
            dummy_config.max_freq = args.max_freq

    # Load shaper_calibrate module if needed
    load_klipper_module(args)

    # Create the graph creator and configure it
    graph_creator = configure_graph_creator(graph_type, args, dummy_config)
    output_filepath = Path(args.output)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    graph_creator.define_output_target(output_filepath)

    print(f'Creating {graph_type} graph...')

    # Load measurements
    measurements_manager = MeasurementsManager(10)
    args.files = [Path(f) for f in args.files]
    if args.files[0].suffix == '.csv':
        measurements_manager.load_from_csvs(args.files)
    elif args.files[0].suffix == '.stdata':
        measurements_manager.load_from_stdata(args.files[0])
    else:
        raise ValueError('Only .stdata or legacy Klipper raw accelerometer CSV files are supported!')

    # Create graph
    graph_creator.create_graph(measurements_manager)

    print('...done!')


if __name__ == '__main__':
    os.environ['SHAKETUNE_IN_CLI'] = '1'
    main()
