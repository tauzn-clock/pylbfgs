VERSION = (0, 1)

def get_version(limit=3):
    """Return the version as a human-format string."""
    return '.'.join([str(i) for i in VERSION[:limit]])

version = get_version()
