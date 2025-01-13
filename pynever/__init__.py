import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_precision_config():
    # DEBUG
    print('Reading precision configuration file...')

    with open(f'{pathlib.Path(__file__).parent.resolve()}/config/precision.txt', 'r') as precision_file:
        precision_config = int(precision_file.read())
        return precision_config

MP_PRECISION = get_precision_config()
print(f'mp.dps is set to {MP_PRECISION}')
