# palgen NES
# Copyright (C) 2023 Persune
# inspired by PalGen, Copyright (C) 2018 DragWx <https://github.com/DragWx>
# testing out the concepts from https://www.nesdev.org/wiki/NTSC_video#Composite_decoding
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser(
    description="yet another NES palette generator",
    epilog="version 0.0.0")
parser.add_argument("-o", "--output", type=str, help=".pal file output")
parser.add_argument("-e", "--emphasis", action="store_true", help="add emphasis entries")
parser.add_argument("-v", "--verbose", action="store_true", help="look at waveforms")
parser.add_argument("-d", "--debug", action="store_true", help="debug messages")

args = parser.parse_args()

# voltage highs and lows
# signal[4][2][2] $0x-$3x, $x0/$xD, no emphasis/emphasis
lidnariq_NESCPU07_2C02G_NES001_signal = np.array([
    [
        [ 0.616, 0.500 ],
        [ 0.228, 0.192 ]
    ],
    [
        [ 0.840, 0.676 ],
        [ 0.312, 0.256 ]
    ],
    [
        [ 1.100, 0.896 ],
        [ 0.552, 0.448 ]
    ],
    [
        [ 1.100, 0.896 ],
        [ 0.880, 0.712 ]
    ]
], np.float64)

# signal buffer normalization
signal_black_point = lidnariq_NESCPU07_2C02G_NES001_signal[1, 1, 0]
signal_white_point = lidnariq_NESCPU07_2C02G_NES001_signal[3, 0, 0]
amplification_factor = 1/(signal_white_point - signal_black_point)

# B-Y and R-Y reduction factors
BY_rf = 1/2.03
RY_rf = 1/1.14

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)

# convert signal RGB to XYZ
RGB_to_XYZ = np.empty([3, 3], np.float64)

# convert XYZ to display RGB
RGB_to_XYZ = np.empty([3, 3], np.float64)

# 111111------
# 22222------2
# 3333------33
# 444------444
# 55------5555
# 6------66666
# ------777777
# -----888888-
# ----999999--
# ---AAAAAA---
# --BBBBBB----
# -CCCCCC-----
# signal buffer for decoding
voltage_buffer = np.empty([12], np.float64)

# decoded YUV buffer,
YUV_buffer = np.empty([8,4,16,3], np.float64)

# decoded RGB buffer
RGB_buffer = np.empty([8,4,16,3], np.float64)

# final 24-bit RGB palette
PaletteColors = np.empty([8,4,16,3], np.uint8)

# settings
settings_brightness = 0

settings_contrast = 0

# hue adjust, in degrees
settings_hue = 0

settings_saturation = 0

# differential phase distortion, in degrees
settings_phase_skew = 5

# fix issue with colors
offset = 2
emphasis_offset = 1
colorburst_phase = 8
colorburst_offset = colorburst_phase - 7

for emphasis in range(8):
    # emphasis bitmask, travelling from lsb to msb
    emphasis_wave = 0
    if bool(emphasis & 0b001):		# tint R; aligned to color phase C
        emphasis_wave |= 0b000001111110;
    if bool(emphasis & 0b010):		# tint G; aligned to color phase 4
        emphasis_wave |= 0b111000000111;
    if bool(emphasis & 0b100):		# tint B; aligned to color phase 8
        emphasis_wave |= 0b011111100000;

    for luma in range(4):
        for hue in range(16):
            # encode voltages into composite waveform
            for wave_phase in range(12):
                # 0 = waveform high; 1 = waveform low
                n_wave_level = 0
                # 1 = emphasis activate
                emphasis_level = int(bool(emphasis_wave & (1 << ((wave_phase - hue + emphasis_offset) % 12))))

                if (wave_phase >= 6): n_wave_level = 1
                    
                # rows $x0 amd $xD
                if (hue == 0x00): n_wave_level = 0
                if (hue == 0x0D): n_wave_level = 1
                
                #rows $xE-$xF
                if (hue >= 0x0E):
                    voltage_buffer[wave_phase] = lidnariq_NESCPU07_2C02G_NES001_signal[1, 1, 0]
                else:
                    voltage_buffer[(wave_phase - hue + offset) % 12] = lidnariq_NESCPU07_2C02G_NES001_signal[luma, n_wave_level, emphasis_level]
            
            if (args.debug):
                print("${0:02X} emphasis {1:03b}".format((luma<<4 | hue), emphasis) + "\n" + str(voltage_buffer))
            if (args.verbose):
                plt.title("${0:02X} emphasis pattern {1:012b}".format((luma<<4 | hue), emphasis_wave))
                x = np.arange(0,12)
                y = voltage_buffer
                plt.xlabel("Sample count")
                plt.ylabel("Voltage")
                plt.step(x, y, linewidth=0.7)
                plt.tight_layout()
                plt.draw()
                plt.show()
            
            # normalize voltage
            voltage_buffer -= signal_black_point
            voltage_buffer *= amplification_factor
            
            # decode voltage buffer to YUV
            UV_buffer = np.empty([12], np.float64)
            # decode Y
            YUV_buffer[emphasis, luma, hue, 0] = np.average(voltage_buffer)
            
            # decode U
            for t in range(12):
                UV_buffer[t] = voltage_buffer[t] * np.sin(
                    2 * np.pi * (1 / 12) * (t + colorburst_offset) +
                    np.radians(settings_hue) -
                    np.radians(settings_phase_skew * luma)
                    )
            YUV_buffer[emphasis, luma, hue, 1] = np.average(UV_buffer)
            
            # decode V
            for t in range(12):
                UV_buffer[t] = voltage_buffer[t] * np.cos(
                    2 * np.pi * (1 / 12) * (t + colorburst_offset) +
                    np.radians(settings_hue) -
                    np.radians(settings_phase_skew * luma)
                    )
            YUV_buffer[emphasis, luma, hue, 2] = np.average(UV_buffer)

            # decode YUV to RGB
            RGB_buffer[emphasis, luma, hue] = np.matmul(np.linalg.inv(RGB_to_YUV), YUV_buffer[emphasis, luma, hue])
            
            
            # normalize RGB to 0.0-1.0
            # TODO: different clipping methods
            for i in range(3):
                RGB_buffer[emphasis, luma, hue, i] = max(0, min(1, 
                    RGB_buffer[emphasis, luma, hue, i]))

            # convert RGB to display output
            # TODO: color primaries transform from one profile to another
            PaletteColors[emphasis, luma, hue] = RGB_buffer[emphasis, luma, hue] * 0xFF

    if not (args.emphasis):
        print("emphasis skipped")
        break

if (args.emphasis):
    PaletteColorsOut = np.reshape(PaletteColors,(32, 16, 3))
else:
    # crop non-emphasis colors if not enabled
    PaletteColorsOut = PaletteColors[0]

if (type(args.output) != type(None)):
    with open(args.output, mode="wb") as PaletteFile:
        PaletteFile.write(PaletteColorsOut)

# figure plotting for palette preview
# TODO: add more graphs, including CIE graph
plt.title("palette")
plt.imshow(PaletteColorsOut)
plt.tight_layout()
plt.draw()
plt.show()
