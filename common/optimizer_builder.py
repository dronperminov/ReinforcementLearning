from tensorflow.keras.optimizers import SGD, Adam


class OptimizerBuilder:
    @staticmethod
    def build(name: str, learning_rate: float):
        if name == 'sgd':
            return SGD(learning_rate)

        if name == 'adam':
            return Adam(learning_rate)

        raise ValueError("unknown optimizer")