from . import Config, create_graph


def main() -> None:
    options = Config.parse_arguments()
    create_graph(options)


if __name__ == '__main__':
    main()
