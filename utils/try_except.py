"""
Utility class for try-except blocks.
Required by YOLOv5.
"""


class TryExcept:
    """
    Try-except wrapper for operations.
    Required by YOLOv5 to handle graceful failures.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            print(f'Error in {self.func.__name__}: {e}')
            return None