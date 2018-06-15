#!/usr/bin/env python3
# Author: Kushagra Juneja, Ramolla Nikhil Reddy
# Description: Resizes batches of 1920x1080 images into 960x540 using interpolation.
# Manual: ./resize.py /path/to 1234
#           where /path/to is the batch directory and 1234 is the number of images
#           and images are named from 0.png to 1234.png
# Issues: Is single threaded, would be faster if multi, but not required for now. Fuck you, GIL.
#         Cwd cannot contain spaces.
from sys import argv
from PIL import Image

if len(argv) != 3:
	print("Error: Argument length must be 2, not %d" % len(argv[1:]))
	exit()

ImageCount = int(argv[2])
Cwd = argv[1] + "/%d.png"

print("Starting...")
print("Resizing is going to be done from %s to %s" % (Cwd % 0, Cwd % (ImageCount-1)))

x = input("Proceed? (yes/no): ")
while x != "yes":
	if x == "no":
		print("Quitting...")
		exit()
	else:
		print("Invalid response. please try again")
	x = input("Proceed? (yes/no): ")

for i in range(ImageCount):
	img = Image.open(Cwd % i)
	img = img.resize((960, 540), Image.ANTIALIAS)
	img.save(Cwd % i)

print("Success: Resizer has run succesfully")
