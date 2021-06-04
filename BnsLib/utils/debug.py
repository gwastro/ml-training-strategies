class DebugCounter(object):
    """A simple debugging class, that can be called between lines.
    """
    def __init__(self, prefix=None):
        self.counter = 0
        self.prefix = '' if prefix is None else prefix
    
    def __call__(self):
        print(self.prefix + str(self.counter))
        self.counter += 1
        
