import RPi.GPIO as GPIO
import math
import os
from datetime import datetime
from time import sleep

PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.IN)


while True:
    value = 1
    # Loop until we read a 0
    while value:
        value = GPIO.input(PIN)
 
    # Grab the start time of the command
    startTime = datetime.now()

    # Used to buffer the command pulses
    command = []

    # The end of the "command" happens when we read more than
    # a certain number of 1s
    numOnes = 0

    # Used to keep track of transitions from 0 to 1
    previousVal = 0

    while True:

        if value != previousVal:
            # The value has changed, so calculate the length of this run
            now = datetime.now()
            pulseLength = now - startTime
	    startTime = now

	    command.append((previousVal, pulseLength.microseconds))

	if value:
	    numOnes += 1
	else:
	    numOnes = 0

	# 10000 is arbitrary, adjust if necessary
	if numOnes > 200:
            break

	previousVal = value
        value = GPIO.input(PIN)
           
    
    print("----------Start----------")
    for (val, pulse) in command:
        print(str(val)+ " " + str(pulse))
    print("----------End----------\n")

    print("Size of array is " + str(len(command)))
