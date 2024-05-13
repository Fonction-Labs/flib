from os import linesep

def dedent(string: str):
    # From: https://stackoverflow.com/questions/1412374/how-to-remove-extra-indentation-of-python-triple-quoted-multi-line-strings
    """
    Allows to de-indent a triple-quoted string.
    """
    return linesep.join(line.lstrip() for line in string.splitlines())
