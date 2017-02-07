import logging


class FixedSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(FixedSingleton, cls).__call__(*args, **kwargs)
        else:
            # cls._instances[cls].__init__(*args, **kwargs)
            print "%s singleton instance already exists, ignore new args" % \
                  (cls.__name__,)
        return cls._instances[cls]


class ReadOnlyClass(object):
    def __setattr__(self, key, value):
        raise AttributeError(
            'Attempted to write attribute %s of a ReadOnlyClass %s' %
            (key, self.__class__.__name__)
        )

    def __delattr__(self, item):
        raise AttributeError(
            'Attempted to delete attribute %s of a ReadOnlyClass %s' %
            (item, self.__class__.__name__)
        )


class FrozenClass(object):
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError(
                'Attempted to create attribute %s of a FrozenClass %s' %
                (key, self.__class__.__name__)
            )
        object.__setattr__(self, key, value)

    def __delattr__(self, item):
        raise AttributeError(
            'Attempted to delete attribute %s of a FrozenClass %s' %
            (item, self.__class__.__name__)
        )


class UncopyableClass(object):
    def __copy__(self):
        raise LookupError(
            'UncopyableClass %s has no method __copy__' %
            self.__class__.__name__,
        )

    def __deepcopy__(self, memo={}):
        raise LookupError(
            'UncopyableClass %s has no method __deepcopy__' %
            self.__class__.__name__,
        )

class LoggingClass(object):
    def __getstate__(self):
        dct = dict(self.__dict__)
        if '_logger' in dct:
            dct['_logger'] = dct['_logger'].name
        return dct

    def __setstate__(self, state):
        if '_logger' in state:
            state['_logger'] = logging.getLogger(state['_logger'])
        self.__dict__.update(state)
