# TCannyMod - Canny edge detection filter for Avisynth2.6.0 / Avisynth+

	TCannyMod is an Avisynth filter plugin rewritten from scratch to
	Avisynth2.6 / AviSynth+ based on tcanny written by Kevin Stone(a.k.a. tritical).


### Syntax:
```
TCannyMod(clip, float "t_h", float "t_l", string "operator", float "scale",
		  float "sigma", bool "strict", int "chroma", int "opt", bool "debug")
```

	- info:
		Builds an edge map using canny edge detection.

	- parameters:

		- clip: planar formats only.

		- t_l: low gradient magnitude threshold for hysteresis (default = 1.0)

		- t_h: high gradient magnitude threshold for hysteresis (default = 8.0)

		- operator: specify operator for edge detection. (default = "standard")
			"standard": use "0 1 0" operator.
			"sobel": use "1 2 1" operator.
			"X Y Z": use "X Y Z" operator.

			               [X,  Y,  Z,              [-X, 0, X,
			"X Y Z" means   0,  0,  0,  for gy and   -Y, 0, Y, for gx.
			               -X, -Y, -Z]               -Z, 0, Z]

			X, Y and Z must be floats or integers.

		- scale: scaling value for gradient magnitude. (default = 1.0)

		- sigma: standard deviation of gaussian blur.
				0 means not bluring before edge detection.
				(0 <= sigma, default = 1.5)

		- strict: How to calculate gradient magnitude.
			true: sqrt(gx^2 + gy^2)
			false: abs(gx) + abs(gy)
			true is a bit slower than false. (default = true)

		- chroma: processing of chroma (default = 0)
			0 - not processing.
			1 - full processing
			2 - copy from source.
			3 - fill with 0x80(128). output is grayscale.
			4 - fill with 0.

		- opt: specify which CPU optimization are used (default = auto.)
			 0: forth NO SIMD optimization routine.
			 1: forth SSE4.1 + SSE2 + SSE routine.
			 2: forth AVX2 + FMA3 + AVX routine.
			 others: use AVX512 routine.

		- debug: append debug information to each frame as frame properties.
				procTime is the time (in microseconds) spent processing the main loop for that frame.



```
GBlur2(clip, float "sigma", int "chroma", int "opt", bool "debug")
```
	- info:
		Gaussian blur filter.

	- parameters:

		- clip: same as TCannyMod.

		- sigma: same as TCannyMod. (default = 0.5)

		- chroma: same as TCannyMod. (default = 0)

		- opt: same as TCannyMod. (default = auto)

		- debug: same as TCannyMod. (default = false)


```
EMask(clip, string "operator", float "scale", float "sigma", int "chroma",
	 int "opt", bool "debug")
```
	- info:
		Generate gradient magnitude edge map.

	- parameters:

		- clip: same as TCannyMod.

		- operator: same as TCannyMod. (default = "standard")

		- scale: same as TCannyMod. (default = 5.1)

		- sigma: same as TCannyMod. (default = 1.5)

		- strict: same as TCannyMod. (default = false)

		- chroma: same as TCannyMod. (default = 1)

		- opt: same as TCannyMod. (default = auto)

		- debug: same as TCannyMod. (default = false)

### Note:
	- TCannyMod requires appropriate memory alignments.
	  Thus, if you want to crop the left side of your source clip before this filter,
	  you have to set crop(align=true).

### Requirements:
	- Avisynth2.6.0/Avisynth+3.7.3 or greater.
	- Windows 7 sp1 or later.
	- Visual C++ Redistributable package
	- AVX capable CPU

### Changelog:
	1.0.0 (20160326):
		- Almost rewrite.
		- VS2013 to VS2015.
		- Add AVX2(64bit only)/SSE4.1(both 32bit and 64bit) support.
		- Change direction values from 1,3,7,15 to 31,63,127,255.
		- Reduce waste processes.

	1.1.0 (20160328):
		- Add EMask().
		- Implement simd non-maximum-suppression.
		- a bit optimized gaussian-blur/hysteresis.

	1.1.1 (20160330):
		- Add AVX2 support for 32bit.

	1.2.0 (2016):
		- Set filter mode as MT_NICE_FILTER on Avisynth+ MT.
		- Use buffer pool on Avisynth+ MT.
		- Disable AVX2/FMA3/AVX code when /arch:AVX2 is not set.
		- Disable AVX2/FMA3/AVX code on Avisynth2.6.

	1.3.0 (20160705)
		- Update avisynth.h to Avisynth+MT r2005.

	1.4.0 (20251216)
		- Update avisynth.h to Avisynth+ 3.7.5,
		- Remove Not AVX capable CPU support.

	2.0.0 (20260102)
		- rewrite to support AVX512 routine.

Source code:

	https://github.com/chikuzen/TCannyMod/


Author  Oka Motofumi (chikuzen.mo at gmail dot com)
