'''
Contains utility functions for better hacker typer
'''

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
    screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

    def reset(self):
        try:
            self.impl.reset()
        except:
            raise NotImplementedError 
    
class _GetchUnix:
    def __init__(self):
        import tty, sys
        import sys, tty, termios
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        
    def __call__(self):
        #try:
        import sys, tty, termios
        #try:
        #self.fd = sys.stdin.fileno()
        #self.old_settings = termios.tcgetattr(self.fd)
        if True:
        #try:
            tty.setcbreak(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        #finally:
        #    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        return ch

    def reset(self):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

from contextlib import contextmanager

@contextmanager
def build_getch():
    getch = _Getch()
    yield getch
    try: # Close for 
        getch.reset()
    except NotImplementedError as e:
        pass
        
    

getch = _Getch()

