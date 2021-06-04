from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Progbar

class ValidationProgbar(Callback):
    def __init__(self, validation_generator):
        self.generator = validation_generator
        self.bar = None
    
    def on_test_begin(self, logs=None):
        print('')
        self.bar = Progbar(len(self.generator))
    
    def on_test_batch_end(self, batch, logs=None):
        if logs is not None:
            print_vals = [('val_' + key, val) for (key, val) in logs.items()]
            self.bar.update(batch+1, values=print_vals)
