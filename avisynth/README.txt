TCannyMod - Canny edge detection filter for Avisynth2.6.0 / Avisynth+


TCannyMod is an Avisynth filter plugin rewritten from scratch to Avisynth2.6
based on tcanny written by Kevin Stone(a.k.a. tritical).


Syntax:

    TCannyMod(clip, int "mode", float "sigma", float "t_h", float "t_l",
              bool "sobel", int "chroma", float gmmax)

    info:
        Builds an edge map using canny edge detection.

    parameters:

        clip - planar formats only.

        mode - sets output format (default = 0)

            0 - thresholded edge map (255 for edge, 0 for non-edge)

            1 - gradient magnitude map.

            2 - edge pixel only gradient direction map (non-edge pixels set to 0)

            3 - gradient direction map

                Gradient direction are normarized to 31, 63, 127 and 255.
                   31 = horizontal
                   63 = 45' up
                  127 = vertical
                  255 = 45' down

            4 - Gaussian blured frame.

        sigma - standard deviation of gaussian blur. 
                0 means not bluring before edge detection.
                (0 <= sigma <= 2.83, default = 1.5)

        t_h - high gradient magnitude threshold for hysteresis (default = 8.0)

        t_l - low gradient magnitude threshold for hysteresis (default = 1.0)

        sobel - use Sobel operator instead of [1, 0, -1] for edge detection. (default = false)

        chroma - processing of chroma (default = 0)

            0 - not processing.

            1 - full processing

            2 - copy from source.

            3 - fill with 0x80(128). output is grayscale.

            4 - fill with 0.

        gmmax - used for scaling gradient magnitude into [0,255] for mode=1 (default = 255)
                 gmmax is internally set to 1.0 if you set it to < 1.0.

        opt - specify which CPU optimization are used (default = auto.)

             0 - forth SSE2 + SSE routine.

             1 - forth SSE4.1 + SSE2 + SSE routine.

             others(64bit only) - use AVX2 + FMA3 + AVX routine.

 ------------------------------------------------------------------------

    GBlur(clip, float "sigma", int "chroma")

    info:
        Gaussian blur filter. Just an alias of TCannyMod(mode=4).

    parameters:

        clip - same as TCannyMod.

        sigma - same as TCannyMod. (default = 0.5)

        chroma - same as TCannyMod. (default = 1)

        opt - same as TCannyMod. (default = -1)


Note:

    TCannyMod requires appropriate memry alignments.
    Thus, if you want to crop the left side of your source clip before this filter,
    you have to set crop(align=true).


Requirements:

    Avisynth2.6.0/Avisynth+r1576 or greater.
    Microsoft Visual C++ 2015 Redistributable Package
    WindowsVistaSP2 or later
    Avisynth2.60 or Avisynth+
    WindowsVistasp2/7sp1/8.1/10
    SSE2 capable CPU


Changelog:

    1.0.0 (20160326):
        - Almost rewrite.
        - VS2013 to VS2015.
        - Add AVX2(64bit only) / SSE4.1(both 32bit and 64bit) support.
        - Change direction values from 1,3,7,15 to 31,63,127,255.
        - Reduce waste processes.

Source code:

    https://github.com/chikuzen/TCannyMod/


Author  Oka Motofumi (chikuzen.mo at gmail dot com)
