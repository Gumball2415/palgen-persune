# Palette Generator

yet another NES palette generator, in Python
<img src="docs/palette sequence.gif">
<img src="docs/waveform sequence.gif">
<img src="docs/QAM sequence.gif">

something to note: there _is_ no one true NES palette, but this generator can pretty much approach colors that looks good enough.

## Requirements

This script requires `numpy` for arrays and matrix math.

This script requires the `colour-science` library for linear light functions, color adaptation, and CIE 1931 colorimetry diagrams.

This script requires `matplotlib` for graphs.

## Usage
```
usage: palgen-persune.py [-h] [-o OUTPUT] [-e] [-d] [-n] [-c] [-w] [-p] [-r] [-s] [-bri BRIGHTNESS] [-con CONTRAST] [-hue HUE] [-sat SATURATION]
                         [-phs PHASE_SKEW] [-blp BLACK_POINT] [-whp WHITE_POINT] [-rfc REFERENCE_COLORSPACE] [-dsc DISPLAY_COLORSPACE]
                         [-cat CHROMATIC_ADAPTATION_TRANSFORM] [-rpr REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R]
                         [-rpg REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G] [-rpb REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B]
                         [-rpw REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W] [-dpr DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R]
                         [-dpg DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G] [-dpb DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B]
                         [-dpw DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W]

yet another NES palette generator

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        .pal file output
  -e, --emphasis        add emphasis entries
  -d, --debug           debug messages
  -n, --normalize       normalize decoded colors within range of RGB (ignores black and white points, contrast, and brightness)
  -c, --clip-black      clip negative values in --normalize function
  -w, --waveforms       view composite waveforms
  -p, --phase-QAM       view QAM demodulation
  -r, --render-png      render views as .pngs in docs folder
  -s, --setup-disable   normalize NES signal levels within luma range (ignores black and white points)
  -bri BRIGHTNESS, --brightness BRIGHTNESS
                        brightness delta, -1.0 to 1.0, default = 0.0
  -con CONTRAST, --contrast CONTRAST
                        contrast delta, 0.0 to 1.0, default = 0.0
  -hue HUE, --hue HUE   hue angle delta, in degrees, default = 0.0
  -sat SATURATION, --saturation SATURATION
                        saturation delta, -1.0 to 1.0, default = 0.0
  -phs PHASE_SKEW, --phase-skew PHASE_SKEW
                        differential phase distortion, in degrees, default = 0.0
  -blp BLACK_POINT, --black-point BLACK_POINT
                        black point, in voltage units relative to blanking, default = 7.5/140.0 (7.5 IRE)
  -whp WHITE_POINT, --white-point WHITE_POINT
                        white point, in voltage units relative to blanking, default = 1.1V (luma level $20)
  -rfc REFERENCE_COLORSPACE, --reference-colorspace REFERENCE_COLORSPACE
                        use colour.RGB_COLOURSPACES reference colorspace, default = "NTSC (1953)"
  -dsc DISPLAY_COLORSPACE, --display-colorspace DISPLAY_COLORSPACE
                        Use colour.RGB_COLOURSPACES display colorspace, default = "ITU-R BT.709"
  -cat CHROMATIC_ADAPTATION_TRANSFORM, --chromatic-adaptation-transform CHROMATIC_ADAPTATION_TRANSFORM
                        chromatic adaptation transform method, default = "XYZ Scaling"
  -rpr REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R, --reference-primaries-r REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R
                        set custom reference color primary R, in CIE xy chromaticity coordinates
  -rpg REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G, --reference-primaries-g REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G
                        set custom reference color primary G, in CIE xy chromaticity coordinates
  -rpb REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B, --reference-primaries-b REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B
                        set custom reference color primary B, in CIE xy chromaticity coordinates
  -rpw REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W, --reference-primaries-w REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W
                        set custom reference whitepoint, in CIE xy chromaticity coordinates
  -dpr DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R, --display-primaries-r DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R
                        set custom display color primary R, in CIE xy chromaticity coordinates
  -dpg DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G, --display-primaries-g DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G
                        set custom display color primary G, in CIE xy chromaticity coordinates
  -dpb DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B, --display-primaries-b DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B
                        set custom display color primary B, in CIE xy chromaticity coordinates
  -dpw DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W, --display-primaries-w DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W
                        set custom display whitepoint, in CIE xy chromaticity coordinates

version 0.3.0
```

## License

This work is licensed under the MIT-0 license.

Copyright (C) Persune 2023.

## Credits

Special thanks to:
- NewRisingSun
- lidnariq
- _aitchFactor
- jekuthiel

This would have not been possible without their help!
