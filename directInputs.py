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

# Commented out section is for windows. Linux appears to be more concise
"""
SendInput = ctypes.cdll.user32.SendInput
# C struct redefinitions
PUL= ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actual Functions

def PressKey(hexKeyCode):
    extra = ctype.c_ulong(0)
    ii_ = Input_I()
    ii.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctype.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.cdll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.cdll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)
"""
