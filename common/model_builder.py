from typing import List, Union, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input


class ModelBuilder:
    @staticmethod
    def build_layer(config: dict, params: Optional[dict] = None):
        if params is None:
            params = {}

        if config['type'] == 'input':
            inputs = config['inputs']
            params['shape'] = (inputs,) if isinstance(inputs, int) else (*inputs,)
            return Input(**params)

        if config['type'] == 'dense':
            params['units'] = config['size']
            params['activation'] = config.get('activation', 'linear')
            params['kernel_initializer'] = 'he_uniform'
            return Dense(**params)

        if config['type'] == 'conv':
            params['filters'] = config['fc']
            params['kernel_size'] = config['fs']
            params['strides'] = config.get('stride', 1)
            params['padding'] = config.get('padding', 'valid')
            params['activation'] = config.get('activation', 'relu')
            params['kernel_initializer'] = 'he_uniform'
            return Conv2D(**params)

        if config['type'] == 'maxpool':
            params['pool_size'] = config['scale']
            return MaxPooling2D(**params)

        if config['type'] == 'flatten':
            return Flatten(**params)

        raise ValueError(f'Invalid layer type "{config["type"]}"')

    @staticmethod
    def build(inputs: Union[int, list], architecture: List[dict]) -> Sequential:
        model = Sequential()
        is_first = True

        for config in architecture:
            params = {}

            if is_first:
                params['input_shape'] = (inputs,) if isinstance(inputs, int) else (*inputs,)
                is_first = False

            layer = ModelBuilder.build_layer(config, params)
            model.add(layer)

        return model
