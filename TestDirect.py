from directInputs import SendInput, SendKeyPress, SendKeyRelease
import time


def Straight():
    SendKeyPress('W')
    SendKeyRelease('S')
    SendKeyRelease('A')
    SendKeyRelease('D')

def Jump():
    SendKeyPress(' ')

time.sleep(5)
Jump()
