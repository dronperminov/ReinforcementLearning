from typing import List, Union
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


class ModelBuilder:
    @staticmethod
    def build(inputs: Union[int, list], architecture: List[dict]) -> Sequential:
        model = Sequential()
        is_first = True

        for config in architecture:
            params = {}

            if is_first:
                params['input_shape'] = (inputs,) if isinstance(inputs, int) else (*inputs,)
                is_first = False

            if config['type'] == 'dense':
                params['units'] = config['size']
                params['activation'] = config.get('activation', 'linear')
                layer = Dense(**params)
            elif config['type'] == 'conv':
                params['filters'] = config['fc']
                params['kernel_size'] = config['fs']
                params['strides'] = config['stride']
                params['padding'] = config['padding']
                params['activation'] = config.get('activation', 'relu')
                layer = Conv2D(**params)
            elif config['type'] == 'maxpool':
                params['pool_size'] = config['scale']
                layer = MaxPooling2D(**params)
            elif config['type'] == 'flatten':
                layer = Flatten(**params)
            else:
                raise ValueError(f'Invalid layer type "{config["type"]}"')

            model.add(layer)

        return model
