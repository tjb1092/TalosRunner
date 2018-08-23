import ctypes

Xtst = ctypes.CDLL("libXtst.so.6")
Xlib = ctypes.CDLL("libX11.so.6")
dpy = Xtst.XOpenDisplay(None)
def SendInput(txt):
    for c in txt:
        sym = Xlib.XStringToKeysym(c)
        code = Xlib.XKeysymToKeycode(dpy, sym)
        Xtst.XTestFakeKeyEvent(dpy, code, True, 0)
        Xtst.XTestFakeKeyEvent(dpy, code, False, 0)
        Xlib.XFlush(dpy)

def SendKeyPress(hexKeyCode):
    sym = Xlib.XStringToKeysym(str(hexKeyCode))
    code = Xlib.XKeysymToKeycode(dpy, sym)
    Xtst.XTestFakeKeyEvent(dpy, code, True, 0)
    Xlib.XFlush(dpy)

def SendKeyRelease(hexKeyCode):
    sym = Xlib.XStringToKeysym(str(hexKeyCode))
    code = Xlib.XKeysymToKeycode(dpy, sym)
    Xtst.XTestFakeKeyEvent(dpy, code, False, 0)
    Xlib.XFlush(dpy)

###############################################################################

def forwards_body():
    SendKeyPress('w')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('d')

def backwards_body():
    SendKeyPress('s')
    SendKeyRelease('w')
    SendKeyRelease('a')
    SendKeyRelease('d')

def left_body():
    SendKeyPress('a')
    SendKeyRelease('s')
    SendKeyRelease('w')
    SendKeyRelease('d')

def right_body():
    SendKeyPress('d')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('w')

def stop_body():
    SendKeyRelease('w')
    SendKeyRelease('s')
    SendKeyRelease('a')
    SendKeyRelease('d')

################################################################################

def up_head():
    SendKeyPress('i')
    SendKeyRelease('j')
    SendKeyRelease('k')
    SendKeyRelease('l')

def down_head():
    SendKeyPress('k')
    SendKeyRelease('j')
    SendKeyRelease('i')
    SendKeyRelease('l')

def left_head():
    SendKeyPress('j')
    SendKeyRelease('i')
    SendKeyRelease('k')
    SendKeyRelease('l')

def right_head():
    SendKeyPress('l')
    SendKeyRelease('j')
    SendKeyRelease('k')
    SendKeyRelease('i')

def stop_head():
    SendKeyRelease('i')
    SendKeyRelease('j')
    SendKeyRelease('k')
    SendKeyRelease('l')
