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
 
    # Loop until we read a 1, which I want to do in my case
    while value:
	value = GPIO.input(PIN)

    # Grab the start time of the command
    startTime = datetime.now()

    # Used to buffer the command pulses
    command = []

    # The end of the "command" happens when we read more than
    # a certain number of 0s.
    numZ = 0

    # Used to keep track of transitions from 0 to 1
    # Needs to be modified if value starts at 1
    previousVal = 1

    while True:

        if value != previousVal:
            # If no change (at beginning), no measure of time
            now = datetime.now()
            pulseLength = now - startTime
	    startTime = now

	    command.append((previousVal, pulseLength.microseconds))

	if value:
	    numZ = 0
	else:
	    numZ += 1

	# 10000 is arbitrary, adjust if necessary
	if numZ > 200:
            break

	previousVal = value
        value = GPIO.input(PIN)   
    
    print("----------Start----------")
    for (val, pulse) in command:
        print(str(val)+ " " + str(pulse))
    print("----------End----------\n")

    print("Size of array is " + str(len(command)))
