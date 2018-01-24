#jfgarces Jan 2018
import RPi.GPIO as GPIO
import time
import argparse
PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

def Signal_Construction(mode, level, wind, moon, temp):

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.OUT)
    
    #duration in seconds
    #constant for debbugging purposes
    constant = 1
    initial_pulse = 0.009*constant
    initial_silence = 0.0045*constant
    base_pulse = 0.00088*constant
    base_silence = 0.00031*constant
    semi_long_silence = 0.00145*constant

    mode_byte = []
    level_byte = []
    wind_byte = []
    moon_byte = []
    temp_byte = []
    final_byte = []

    #beginning of signal in seconds
    beg = [initial_pulse, initial_silence]
    
    # value of bit 0 = [base_silence, semi_long_pulse], bit 1 = [base_silence, base_pulse]
    bit_zero = [base_pulse, base_silence]
    bit_one = [base_pulse, semi_long_silence]

    #Signal Construction
    
    signal = []
    signal.extend(beg)
    
    #signal extension of mode byte   
    mode_byte = Get_mode(mode)
    for i in mode_byte:
        if i == 0:
            signal.extend(bit_zero)
        else:
            signal.extend(bit_one)

    #signal extension of level byte
    level_byte = Get_level(level)
    for x in level_byte:
        if x == 0:
            signal.extend(bit_zero)
        else:
            signal.extend(bit_one)

    #signal extension of wind byte
    if wind == 0:
        signal.extend(bit_zero)
    else:
        signal.extend(bit_one)

    #signal extension of moon byte
    if moon == 0:
        signal.extend(bit_zero)
    else:
        signal.extend(bit_one)

    #signal extension of temp byte
    temp_byte = Get_temp(temp)
    for i in temp_byte:
        if i == 0:
            signal.extend(bit_zero)
        else:
            signal.extend(bit_one)

    #signal extension of end of signal
    final_byte = Get_final()
    for i in final_byte:
        if i == 0:
            signal.extend(bit_zero)
        else:
            signal.extend(bit_one)

    #signal extension of half bit, just bit_one
    #signal.extend(bit_one)   

    #Signal emission, not bits or bytes

    #starting time set up:       
    start = time.time()

    #Signal emission
    for i in range(0,len(signal)):
        now = start
        if i%2 == 0:
            #high voltage while bit in signal[i] is odd number
            GPIO.output(PIN, 1)
            while now < start + signal[i]:
                now = time.time()
            start += signal[i]
        else:
            #low voltage while bit in signal[i] is even number
            GPIO.output(PIN, 0)
            while now < start + signal[i]:
                now = time.time()
            #we don't reset start point, we keep it as reference
            start += signal[i]

    time.sleep(0.1)        
    GPIO.cleanup()

def Get_mode(mode):

    byte = []
    if mode == "off_mode":
        byte = [0,0,0,0]
    elif mode == "no_mode":
        byte = [0,0,0,1]
    elif mode == "winter_mode":
            byte = [1,0,0,1]
    elif mode == "bath_mode":
        byte = [0,1,0,1]
    elif mode == "fan_mode":
        byte = [1,1,0,1]
    elif mode == "summer_mode":
        byte = [0,0,1,1]

    return byte 

def Get_level(level):

    byte = []
    if level == 0:
        byte = [0,0]
    elif level == 1:
        byte = [1,0]
    elif level == 2:
        byte = [0,1]
    elif level == 3:
        byte = [1,1]
        
    return byte

def Get_temp(temp):

    byte = []
    if temp == 16:
        byte = [0,0,0,0]
    elif temp == 17:
        byte = [1,0,0,0]
    elif temp == 18:
        byte = [0,1,0,0]
    elif temp == 19:
        byte = [1,1,0,0]
    elif temp == 20:
        byte = [0,0,1,0]
    elif temp == 21:
        byte = [1,0,1,0]
    elif temp == 22:
        byte = [0,1,1,0]
    elif temp == 23:
        byte = [1,1,1,0]
    elif temp == 24:
        byte = [0,0,0,1]
    elif temp == 25:
        byte = [1,0,0,1]
    elif temp == 26:
        byte = [0,1,0,1]
    elif temp == 27:
        byte = [1,1,0,1]
    elif temp == 28:
        byte = [0,0,1,1]
    elif temp == 29:
        byte = [1,0,1,1]
    elif temp == 30:
        byte = [0,1,1,1]

    return byte

def Get_final():

    byte = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1]

    return byte


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AC setup')
    parser.add_argument('-lp','--loop', metavar = 'program', type = int, required = True, help = 'times to repeat the IR Signal')
    parser.add_argument('-m','--mode', metavar = 'setup', required = True, help = 'AC mode setup. winter_mode, summer_mode, bath_mode, fan_mode, no_mode, off_mode')
    parser.add_argument('-l','--level', metavar = 'setup', type = int, required = True, help = 'Fan speed setup,0 to 3')
    parser.add_argument('-w','--wind', metavar = 'setup', type = int, required = True, help = 'Wind setup, not sure what is it for, 0 off, 1 on')
    parser.add_argument('-mn','--moon', metavar = 'setup', type = int, required = True, help = 'Night Mode setup, 0 off, 1 on')
    parser.add_argument('-t','--temp', metavar = 'setup', type = int, required = True, help = 'Temperature setup, 16 to 30 C')
    args = parser.parse_args()

    for i in range(0,args.loop):
        Signal_Construction(mode=args.mode,level=args.level,wind=args.wind,moon=args.moon,temp=args.temp)

