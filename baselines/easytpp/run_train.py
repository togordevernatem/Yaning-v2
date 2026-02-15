import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Path to yaml config file.')
    parser.add_argument('--experiment_id', type=str, required=True,
                        help='Experiment id in the config file.')
    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    runner = Runner.build_from_config(config)
    runner.run()

if __name__ == '__main__':
    main()
