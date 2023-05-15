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
import colour.plotting.diagrams

parser=argparse.ArgumentParser(
    description="yet another NES palette generator",
    epilog="version 0.3.0")
parser.add_argument("-o", "--output", type=str, help=".pal file output")
parser.add_argument("-e", "--emphasis", action="store_true", help="add emphasis entries")
parser.add_argument("-d", "--debug", action="store_true", help="debug messages")
parser.add_argument("-n", "--normalize", action="store_true", help="normalize decoded colors within range of RGB (ignores black and white points, contrast, and brightness)")
parser.add_argument("-c", "--clip-black", action="store_true", help="clip negative values in --normalize function")
parser.add_argument("-w", "--waveforms", action="store_true", help="view composite waveforms")
parser.add_argument("-p", "--phase-QAM", action="store_true", help="view QAM demodulation")
parser.add_argument("-r", "--render-png", action="store_true", help="render views as .pngs in docs folder")
parser.add_argument("-s", "--setup-disable", action="store_true", help="normalize NES signal levels within luma range (ignores black and white points)")

parser.add_argument(
    "-bri",
    "--brightness",
    type = np.float64,
    help = "brightness delta, -1.0 to 1.0, default = 0.0",
    default = 0.0)
parser.add_argument(
    "-con",
    "--contrast",
    type = np.float64,
    help = "contrast delta, 0.0 to 1.0, default = 0.0",
    default = 0.0)

parser.add_argument(
    "-hue",
    "--hue",
    type = np.float64,
    help = "hue angle delta, in degrees, default = 0.0",
    default = 0)
parser.add_argument(
    "-sat",
    "--saturation",
    type = np.float64,
    help ="saturation delta, -1.0 to 1.0, default = 0.0",
    default = 0)
parser.add_argument(
    "-phs",
    "--phase-skew",
    type = np.float64,
    help = "differential phase distortion, in degrees, default = 0.0",
    default = 0)
parser.add_argument(
    "-blp",
    "--black-point",
    type = np.float64,
    help = "black point, in voltage units relative to blanking, default = 7.5/140.0 (7.5 IRE)",
    default =  7.5/140)
parser.add_argument(
    "-whp",
    "--white-point",
    type = np.float64,
    help = "white point, in voltage units relative to blanking, default = 1.1V (luma level $20)")

parser.add_argument(
    "-rfc",
    "--reference-colorspace",
    type = str,
    help = "use colour.RGB_COLOURSPACES reference colorspace, default = \"NTSC (1953)\"",
    default = 'NTSC (1953)')
parser.add_argument(
    "-dsc",
    "--display-colorspace",
    type = str,
    help = "Use colour.RGB_COLOURSPACES display colorspace, default = \"ITU-R BT.709\"",
    default = 'ITU-R BT.709')
parser.add_argument(
    "-cat",
    "--chromatic-adaptation-transform",
    type = str,
    help = "chromatic adaptation transform method, default = \"XYZ Scaling\"",
    default = 'XYZ Scaling')

parser.add_argument(
    "-rpr",
    "--reference-primaries-r",
    type = np.float64,
    nargs=2,
    help = "set custom reference color primary R, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-rpg",
    "--reference-primaries-g",
    type = np.float64,
    nargs=2,
    help = "set custom reference color primary G, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-rpb",
    "--reference-primaries-b",
    type = np.float64,
    nargs=2,
    help = "set custom reference color primary B, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-rpw",
    "--reference-primaries-w",
    type = np.float64,
    nargs=2,
    help = "set custom reference whitepoint, in CIE xy chromaticity coordinates")

parser.add_argument(
    "-dpr",
    "--display-primaries-r",
    type = np.float64,
    nargs=2,
    help = "set custom display color primary R, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-dpg",
    "--display-primaries-g",
    type = np.float64,
    nargs=2,
    help = "set custom display color primary G, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-dpb",
    "--display-primaries-b",
    type = np.float64,
    nargs=2,
    help = "set custom display color primary B, in CIE xy chromaticity coordinates")
parser.add_argument(
    "-dpw",
    "--display-primaries-w",
    type = np.float64,
    nargs=2,
    help = "set custom display whitepoint, in CIE xy chromaticity coordinates")

args = parser.parse_args()

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

# special thanks to NewRisingSun for teaching me how chromatic adaptations work!
# special thanks to _aitchFactor for pointing out that colour-science has
# chromatic adaptation functions!

# reference color profile colorspace
s_colorspace = colour.RGB_COLOURSPACES[args.reference_colorspace].copy()
if (type(args.reference_primaries_r) != type(None)) and (type(args.reference_primaries_g) != type(None)) and (type(args.reference_primaries_b) != type(None)): 
    s_colorspace.name = "custom primaries"
    s_colorspace.primaries = np.array([
        args.reference_primaries_r,
        args.reference_primaries_g,
        args.reference_primaries_b
    ])
if (type(args.reference_primaries_w) != type(None)):
    s_colorspace.whitepoint = args.reference_primaries_w
    s_colorspace.whitepoint_name = "custom whitepoint"

# display color profile colorspace
t_colorspace = colour.RGB_COLOURSPACES[args.display_colorspace].copy()
if (type(args.display_primaries_r) != type(None)) and (type(args.display_primaries_g) != type(None)) and (type(args.display_primaries_b) != type(None)):
    t_colorspace.name = "custom primaries"
    t_colorspace.primaries = np.array([
        args.display_primaries_r,
        args.display_primaries_g,
        args.display_primaries_b
    ])
if (type(args.display_primaries_w) != type(None)):
    t_colorspace.whitepoint = args.display_primaries_w
    t_colorspace.whitepoint_name = "custom whitepoint"

s_colorspace.name = "Reference colorspace: {}".format(s_colorspace.name)
t_colorspace.name = "Display colorspace: {}".format(t_colorspace.name)

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
U_buffer = np.empty([12], np.float64)
V_buffer = np.empty([12], np.float64)

# decoded YUV buffer,
YUV_buffer = np.empty([8,4,16,3], np.float64)

# decoded RGB buffer
# has to be zero'd out for the normalize function to work
RGB_buffer = np.zeros([8,4,16,3], np.float64)

# fix issue with colors
offset = 1
emphasis_offset = 1
colorburst_phase = 8

# due to the way the waveform is encoded, the hue is off by 15 degrees,
# or 1/2 of a sample
colorburst_offset = colorburst_phase - 6 - 0.5

# signal buffer normalization
if (args.setup_disable):
    signal_black_point = signal_table[1, 1, 0]
    signal_white_point = signal_table[3, 0, 0]
else:
    signal_black_point = signal_table[1, 1, 0] + args.black_point
    if type(args.white_point) != type(None):
        signal_white_point = signal_table[1, 1, 0] + args.white_point
    else:
        signal_white_point = signal_table[3, 0, 0]

# used for image sequence plotting
sequence_counter = 0

# figure plotting for palette preview
# TODO: interactivity
def NES_palette_plot(RGB_buffer, RGB_raw, emphasis, luma_range, all_emphasis = False, export_image = False):
    if all_emphasis or not args.emphasis:
        RGB_sub = RGB_buffer
        RGB_sub_raw = RGB_raw
    else:
        RGB_sub = np.split(RGB_buffer, 8, 0)[emphasis]
        RGB_sub_raw = np.split(RGB_raw, 8, 0)[emphasis]

    YUV_buffer_out = np.empty([luma_range, 16, 3], np.float64)
    for luma in range(luma_range):
        for hue in range(16):
            YUV_buffer_out[luma, hue] = np.matmul(RGB_to_YUV, RGB_sub[luma, hue])

    color_theta = np.arctan2(YUV_buffer_out[:, :, 2], YUV_buffer_out[:, :, 1])
    color_r = YUV_buffer_out[:, :, 0]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax0 = fig.add_subplot(gs[1, 1])
    ax1 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2 = fig.add_subplot(gs[:, 0])

    fig.suptitle('NES palette')
    fig.tight_layout()

    # colors
    ax0.set_title("Color swatches")
    ax0.imshow(RGB_sub)

    # polar plot
    ax1.set_title("RGB color phase")
    ax1.set_yticklabels([])
    ax1.axis([0, 2*np.pi, 0, 1])
    ax1.scatter(color_theta, color_r, c=np.reshape(RGB_sub,(luma_range*16, 3)), marker=None, s=color_r*500, zorder=3)

    # CIE graph
    colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
        RGB_sub_raw,
        colourspace=s_colorspace,
        show_whitepoints=False,
        scatter_kwargs=dict(c=np.reshape(RGB_sub,(luma_range*16, 3)),alpha=0.1),
        plot_kwargs=dict(color="gray"),
        figure=fig,
        axes=ax2,
        standalone=False,
        show_diagram_colours=False,
        show_spectral_locus=True,
        spectral_locus_colours='RGB',
        transparent_background=False)
    colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
        RGB_sub,
        colourspace=t_colorspace,
        show_whitepoints=False,
        scatter_kwargs=dict(c=np.reshape(RGB_sub,(luma_range*16, 3))),
        plot_kwargs=dict(color="red"),
        figure=fig,
        axes=ax2,
        standalone=False,
        show_diagram_colours=False,
        show_spectral_locus=True,
        spectral_locus_colours='RGB',
        transparent_background=False)
    ax2.set_title("CIE 1931 chromaticity diagram")
    ax2.grid(which='both', color='grey', linewidth=0.5, linestyle='-', alpha=0.2)

    fig.set_size_inches(16, 9)
    if (export_image):
        plt.savefig("docs/palette sequence {0:03}.png".format(emphasis), dpi=120)
    else:
        plt.show()
    plt.close()


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
            if (args.waveforms):
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
                if args.render_png:
                    plt.savefig("docs/waveform sequence {0:03}.png".format(sequence_counter), dpi=120)
                else:
                    plt.show()
                plt.close()

            # normalize voltage

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

            # apply black and white points, brightness, and contrast
            RGB_buffer[emphasis, luma, hue] -= signal_black_point
            RGB_buffer[emphasis, luma, hue] /= (signal_white_point - signal_black_point)
            RGB_buffer[emphasis, luma, hue] += args.brightness
            RGB_buffer[emphasis, luma, hue] *= (args.contrast + 1)

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
                
                range_axis = (signal_white_point / (signal_white_point - signal_black_point)) - signal_black_point
                axY.set_title("Y decoding")
                axY.set_ylabel("value")
                axY.axis([0, 12, 0, range_axis])
                axY.plot(x, voltage_buffer, 'o-', linewidth=0.7, label='composite signal')
                axY.plot(x, np.full((12), Y_avg), 'o-', linewidth=0.7, label='Y value = {:< z.3f}'.format((Y_avg - signal_black_point) / (signal_white_point - signal_black_point)))
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
                if args.render_png:
                    plt.savefig("docs/QAM sequence {0:03}.png".format(sequence_counter), dpi=120)
                else:
                    plt.show()
                plt.close()

            sequence_counter += 1
    if not (args.emphasis):
        break

# convert RGB to display output
RGB_raw = RGB_buffer
# convert signal to linear light
RGB_buffer = colour.models.oetf_inverse_BT709(RGB_buffer)

# transform linear light
RGB_buffer[:, :, :] = colour.RGB_to_RGB(
    RGB_buffer[:, :, :],
    s_colorspace,
    t_colorspace,
    chromatic_adaptation_transform=args.chromatic_adaptation_transform)

# convert linear light to signal
RGB_buffer = colour.models.oetf_BT709(RGB_buffer)

# normalize RGB to 0.0-1.0
# TODO: different clipping methods
if (args.normalize):
    if args.clip_black:
        np.clip(RGB_buffer, 0, None, out=RGB_buffer)
        np.clip(RGB_raw, 0, None, out=RGB_raw)
    RGB_buffer -= np.amin(RGB_buffer)
    RGB_buffer /= (np.amax(RGB_buffer) - np.amin(RGB_buffer))
    RGB_raw -= np.amin(RGB_raw)
    RGB_raw /= (np.amax(RGB_raw) - np.amin(RGB_raw))
else:
    np.clip(RGB_buffer, 0, 1, out=RGB_buffer)
    np.clip(RGB_raw, 0, 1, out=RGB_raw)

# display data about the palette, and optionally write a .pal file
if (args.emphasis):
    RGB_buffer = np.reshape(RGB_buffer,(32, 16, 3))
    RGB_raw = np.reshape(RGB_raw,(32, 16, 3))
    luma_range = 32
else:
    # crop non-emphasis colors if not enabled
    RGB_buffer = RGB_buffer[0]
    RGB_raw = RGB_raw[0]
    luma_range = 4

if (type(args.output) != type(None)):
    with open(args.output, mode="wb") as Palette_file:
        Palette_file.write(np.uint8(RGB_buffer * 0xFF))

if (args.render_png):
    for emphasis in range(8):
        NES_palette_plot(RGB_buffer, RGB_raw, emphasis, 4, False, True)
        if not (args.emphasis):
            break

NES_palette_plot(RGB_buffer, RGB_raw, 0, luma_range, True)
