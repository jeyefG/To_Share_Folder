import RPi.GPIO as GPIO
import time
import argparse
PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

def Signal_Construction():

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.OUT)
    
    #duration in seconds
    #constant for debbugging purposes
    constant = 1
    initial_pulse = 0.00325*constant
    initial_silence = 0.001625*constant
    base_pulse = 0.0008*constant
    base_silence = 0.00004*constant
    semi_long_silence = 0.0008*constant

    raw_byte = [1,1,0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0]

    #beginning of signal in seconds
    beg = [initial_pulse, initial_silence]
    
    # value of bit 0 = [base_silence, semi_long_pulse], bit 1 = [base_silence, base_pulse]
    bit_zero = [base_pulse, base_silence]
    bit_one = [base_pulse, semi_long_silence]

    #Signal Construction
    
    signal = []
    signal.extend(beg)
    
    #signal extension of mode byte   

    for i in raw_byte:
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
            
    GPIO.cleanup()

if __name__ == '__main__':

    Signal_Construction()
    
