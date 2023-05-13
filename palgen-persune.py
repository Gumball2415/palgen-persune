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
import matplotlib.gridspec as gridspec
import colour.models

# voltage highs and lows
# from https://forums.nesdev.org/viewtopic.php?p=159266#p159266
# signal[4][2][2] $0x-$3x, $x0/$xD, no emphasis/emphasis
signal_table = np.array([
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

# B-Y and R-Y reduction factors
BY_rf = 1/2.03
RY_rf = 1/1.14

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)

# reference color profile, in CIE xy chromaticities
reference_profile = np.array([
    [0.670, 0.330],     # red
    [0.210, 0.710],     # green
    [0.140, 0.080],     # blue
    [0.313, 0.329]      # white point
], np.float64)

# display color profile, in CIE xy chromaticities
display_profile = np.array([
    [0.670, 0.330],     # red
    [0.210, 0.710],     # green
    [0.140, 0.080],     # blue
    [0.313, 0.329]      # white point
], np.float64)

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
U_buffer = np.zeros([12], np.float64)
V_buffer = np.zeros([12], np.float64)

# decoded YUV buffer,
YUV_buffer = np.zeros([8,4,16,3], np.float64)

# decoded RGB buffer
RGB_buffer = np.zeros([8,4,16,3], np.float64)

# fix issue with colors
offset = 1
emphasis_offset = 1
colorburst_phase = 8

# due to the way the waveform is encoded, the hue is off by 15 degrees, or 1/2 of a sample
colorburst_offset = colorburst_phase - 6 - 0.5

# plt.style.use('dark_background')

parser=argparse.ArgumentParser(
    description="yet another NES palette generator",
    epilog="version 0.1.2")
parser.add_argument("-o", "--output", type=str, help=".pal file output")
parser.add_argument("-e", "--emphasis", action="store_true", help="add emphasis entries")
parser.add_argument("-d", "--debug", action="store_true", help="debug messages")
parser.add_argument("-n", "--normalize", action="store_true", help="normalize colors within range of RGB")
parser.add_argument("-v", "--visualize-wave", action="store_true", help="render composite waveforms as .png in docs folder")
parser.add_argument("-p", "--phase-QAM", action="store_true", help="render QAM demodulation as .png in docs folder")

parser.add_argument(
    "--brightness",
    type = np.float64,
    help = "brightness, -1.0 to 1.0, default = 0.0",
    default = 0.0)
parser.add_argument(
    "--contrast",
    type = np.float64,
    help = "contrast, 0.0 to 1.0, default = 0.0",
    default = 0.0)

parser.add_argument(
    "--hue",
    type = np.float64,
    help = "hue angle delta, in degrees, default = 0.0",
    default = 0)
parser.add_argument(
    "--saturation",
    type = np.float64,
    help ="saturation delta, -1.0 to 1.0, default = 0.0",
    default = 0)
parser.add_argument(
    "--phase-skew",
    type = np.float64,
    help = "differential phase distortion, in degrees, default = 0.0",
    default = 0)
parser.add_argument(
    "--black-point",
    type = np.float64,
    help = "black point, in voltage units relative to blanking, default = 7.5/140.0",
    default =  7.5/140) # 7.5 IRE
parser.add_argument(
    "--white-point",
    type = np.float64,
    help = "white point, in voltage units relative to blanking, default = 100.0/140.0",
    default = 100/140) # 100 IRE
    
# parser.add_argument(
    # "--white-point",
    # type = np.float64,
    # help = "white point, in voltage units relative to blanking, default = 100.0/140.0",
    # default = "")

args = parser.parse_args()

# signal buffer normalization
signal_black_point = signal_table[1, 1, 0] + args.black_point
signal_white_point = signal_table[1, 1, 0] + args.white_point
amplification_factor = 1/(signal_white_point - signal_black_point)

# used for image sequence plotting
sequence_counter = 0

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
                    voltage_buffer[wave_phase] = signal_table[1, 1, 0]
                else:
                    voltage_buffer[(wave_phase - hue + offset) % 12] = signal_table[luma, n_wave_level, emphasis_level]

            # TODO: better filter voltage buffer

            # convolution approach
            # kernel = np.array([0.25, 0.75, 1, -0.2, -0.05, 0], np.float64)
            # voltage_buffer = np.delete(np.convolve(voltage_buffer, kernel), [0, 13, 14, 15, 16])

            # fft approach
            # voltage_buffer = np.fft.fft(voltage_buffer)
            # kernel = np.array([1, 1, 0.5, 0.5, 0.5, 0.3, 0, 0, 0, 0, 0, 0], np.float64)
            # voltage_buffer *= kernel
            # voltage_buffer = np.fft.ifft(voltage_buffer)

            if (args.debug):
                print("${0:02X} emphasis {1:03b}".format((luma<<4 | hue), emphasis) + "\n" + str(voltage_buffer))
            if (args.visualize_wave):
                fig = plt.figure(tight_layout=True)
                fig.suptitle("${0:02X} emphasis {1:03b}".format((luma<<4 | hue), emphasis))
                ax = fig.subplots()
                x = np.arange(0,12)
                y = voltage_buffer
                ax.axis([0, 12, 0, 1.5])
                ax.set_xlabel("Sample count")
                ax.set_ylabel("Voltage")
                ax.plot(x, y, 'o-', linewidth=0.7)
                
                fig.set_size_inches(16, 9)
                plt.savefig("docs/waveform sequence {0:03}.png".format(sequence_counter), dpi=120)
                plt.close()

            # normalize voltage
            voltage_buffer -= signal_black_point
            voltage_buffer *= amplification_factor

            # decode voltage buffer to YUV
            # decode Y
            YUV_buffer[emphasis, luma, hue, 0] = np.average(voltage_buffer)

            # decode U
            for t in range(12):
                U_buffer[t] = voltage_buffer[t] * np.sin(
                    2 * np.pi / 12 * (t + colorburst_offset) +
                    np.radians(args.hue) -
                    np.radians(args.phase_skew * luma))
            YUV_buffer[emphasis, luma, hue, 1] = np.average(U_buffer) * (args.saturation + 1)

            # decode V
            for t in range(12):
                V_buffer[t] = voltage_buffer[t] * np.cos(
                    2 * np.pi / 12 * (t + colorburst_offset) +
                    np.radians(args.hue) -
                    np.radians(args.phase_skew * luma))
            YUV_buffer[emphasis, luma, hue, 2] = np.average(V_buffer) * (args.saturation + 1)


            # decode YUV to RGB
            RGB_buffer[emphasis, luma, hue] = np.matmul(np.linalg.inv(RGB_to_YUV), YUV_buffer[emphasis, luma, hue])

            # apply brightness and contrast
            RGB_buffer[emphasis, luma, hue] = (RGB_buffer[emphasis, luma, hue] + args.brightness) * (args.contrast + 1)

            # visualize chroma decoding
            if (args.phase_QAM):
                fig = plt.figure(tight_layout=True)
                gs = gridspec.GridSpec(3, 2)

                axY = fig.add_subplot(gs[0,0])
                axU = fig.add_subplot(gs[1,0])
                axV = fig.add_subplot(gs[2,0])
                ax1 = fig.add_subplot(gs[:,1], projection='polar')
                fig.suptitle("QAM demodulating ${0:02X} emphasis {1:03b}".format((luma<<4 | hue), emphasis))
                w = voltage_buffer
                x = np.arange(0,12)
                Y_avg = np.average(voltage_buffer)
                U_avg = np.average(U_buffer)
                V_avg = np.average(V_buffer)
                
                range_axis = (signal_white_point * amplification_factor) - signal_black_point
                axY.set_title("Y decoding")
                axY.set_ylabel("value")
                axY.axis([0, 12, -1*range_axis, range_axis])
                axY.plot(x, voltage_buffer, 'o-', linewidth=0.7, label='normalized signal')
                axY.plot(x, np.full((12), Y_avg), 'o-', linewidth=0.7, label='Y value = {:< z.3f}'.format(Y_avg))
                axY.legend(loc='lower right')
                
                axU.set_title("U decoding")
                axU.set_ylabel("value")
                axU.axis([0, 12, -1*range_axis, range_axis])
                axU.plot(x, U_buffer, 'o-', linewidth=0.7, label='demodulated U signal')
                axU.plot(x, np.full((12), U_avg), 'o-', linewidth=0.7, label='U value = {:< z.3f}'.format(U_avg))
                axU.legend(loc='lower right')
                
                axV.set_title("V decoding")
                axV.set_ylabel("value")
                axV.axis([0, 12, -1*range_axis, range_axis])
                axV.plot(x, V_buffer, 'o-', linewidth=0.7, label='demodulated V signal')
                axV.plot(x, np.full((12), V_avg), 'o-', linewidth=0.7, label='V value = {:< z.3f}'.format(V_avg))
                axV.legend(loc='lower right')
                
                color_theta = np.arctan2(V_avg, U_avg)
                color_r =  np.sqrt(U_avg**2 + V_avg**2)
                ax1.axis([0, 2*np.pi, 0, 0.3])
                ax1.set_title("Phasor plot")
                ax1.scatter(color_theta, color_r)
                ax1.vlines(color_theta, 0, color_r)
                
                fig.set_size_inches(16, 9)
                plt.savefig("docs/QAM sequence {0:03}.png".format(sequence_counter), dpi=120)
                plt.close()

            sequence_counter += 1
    if not (args.emphasis):
        print("emphasis skipped")
        break


# normalize RGB to 0.0-1.0
# TODO: different clipping methods
if args.normalize:
    RGB_buffer -= np.amin(RGB_buffer)
    RGB_buffer /= (np.amax(RGB_buffer) - np.amin(RGB_buffer))
else:
    np.clip(RGB_buffer, 0, 1, out=RGB_buffer)

# convert RGB to display output

# convert signal to linear light
RGB_buffer = colour.models.oetf_inverse_BT709(RGB_buffer)

# transform linear light

# convert linear light to signal
RGB_buffer = colour.models.oetf_BT709(RGB_buffer)


if (args.emphasis):
    Palette_colors_out = np.reshape(RGB_buffer,(32, 16, 3))
    luma_range = 32
else:
    # crop non-emphasis colors if not enabled
    Palette_colors_out = RGB_buffer[0]
    luma_range = 4

YUV_buffer_out = np.empty([luma_range, 16, 3], np.float64)
for luma in range(luma_range):
    for hue in range(16):
        YUV_buffer_out[luma, hue] = np.matmul(RGB_to_YUV, Palette_colors_out[luma, hue])

# display data about the palette, and optionally write a .pal file

if (type(args.output) != type(None)):
    with open(args.output, mode="wb") as Palette_file:
        Palette_file.write(np.uint8(Palette_colors_out * 0xFF))

for emphasis in range(8):
    YUV_subbuffer = np.empty([4, 16, 3], np.float64)
    subpalette_buffer = RGB_buffer[emphasis]
    for luma in range(4):
        for hue in range(16):
            YUV_subbuffer[luma, hue] = np.matmul(RGB_to_YUV, subpalette_buffer[luma, hue])
    color_theta = np.arctan2(YUV_subbuffer[:, :, 2], YUV_subbuffer[:, :, 1])
    color_r = YUV_subbuffer[:, :, 0]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2 = fig.add_subplot(gs[1, 1])

    fig.suptitle('NES palette (emphasis = {:03b})'.format(emphasis))
    fig.tight_layout()
    # colors
    ax0.set_title("Color swatches")
    ax0.imshow(subpalette_buffer)

    # polar plot
    ax1.set_title("RGB color phase")
    ax1.set_yticklabels([])
    ax1.axis([0, 2*np.pi, 0, 1])
    ax1.scatter(color_theta, color_r, c=np.reshape(subpalette_buffer,(4*16, 3)), marker=None, s=color_r*500, zorder=3)

    # CIE graph
    ax2.set_title("CIE graph (todo)")

    fig.set_size_inches(16, 9)
    plt.savefig("docs/palette sequence {0:03}.png".format(emphasis), dpi=120)
    plt.close()
    if not (args.emphasis):
        break

color_theta = np.arctan2(YUV_buffer_out[:, :, 2], YUV_buffer_out[:, :, 1])
color_r = YUV_buffer_out[:, :, 0]

# figure plotting for palette preview
# TODO: interactivity

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1], projection='polar')
ax2 = fig.add_subplot(gs[1, 1])

fig.suptitle('NES palette')
fig.tight_layout()
# colors
ax0.set_title("Color swatches")
ax0.imshow(Palette_colors_out)

# polar plot
ax1.set_title("RGB color phase")
ax1.set_yticklabels([])
ax1.axis([0, 2*np.pi, 0, 1])
ax1.scatter(color_theta, color_r, c=np.reshape(Palette_colors_out,(luma_range*16, 3)), marker=None, s=color_r*500, zorder=3)

# CIE graph
ax2.set_title("CIE graph (todo)")

fig.set_size_inches(16, 9)
plt.show()
