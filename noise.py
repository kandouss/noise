#!/usr/bin/python

import noise_math as nm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class CirclePlot:
	def __init__():
		self.S = nm.StaticNoise()
		self.C = 4

def noise_test():
	n=32
	time_zoom = 2.0
	space_zoom = 10.0
	S = nm.StaticNoise(n=n)
	X = np.arange(n*float(space_zoom))/space_zoom
	for t in range(int(n*time_zoom)):
		time = float(t) / time_zoom
		Y = [5.0+S[x,time] for x in X]
		X2 = X/float(n)*2.*np.pi
		ax = plt.subplot(111, polar=True)
		ax.grid(False)
		ax.plot(X2, Y)
		plt.show()


def main():
	noise_test()

if __name__=="__main__":
	main()

