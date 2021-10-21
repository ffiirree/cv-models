import sys
import functools

__all__ = ['export', 'config']


def export(obj):
    if hasattr(sys.modules[obj.__module__], '__all__'):
        sys.modules[obj.__module__].__all__.append(obj.__name__)
    else:
        sys.modules[obj.__module__].__all__ = [obj.__name__]
    return obj


def config(url='', **settings):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs['url'] = url
            kwargs['arch'] = func.__name__
            return func(*args, **{**settings, **kwargs})
        return wrapper

    return decorator
