# Palette Generator

yet another NES palette generator, in Python
<img src="docs/palette sequence.gif">
<img src="docs/waveform sequence.gif">
<img src="docs/QAM sequence.gif">

something to note: there _is_ no one true NES palette, but this generator can pretty much approach colors that looks good enough.

## Requirements

This script requires `numpy` for arrays and matrix math.

This script requires `colour.models` library for linear light functions and `colour.plotting.diagrams` library for CIE 1931 colorimetry diagrams.

This script requires `matplotlib` for graphs.

## Usage
```
usage: palgen-persune.py [-h] [-o OUTPUT] [-e] [-d] [-n] [-c] [-w] [-p] [-r] [-s] [--brightness BRIGHTNESS] [--contrast CONTRAST] [--hue HUE]
                         [--saturation SATURATION] [--phase-skew PHASE_SKEW] [--black-point BLACK_POINT] [--white-point WHITE_POINT]
                         [--reference-primaries-r REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R]
                         [--reference-primaries-g REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G]
                         [--reference-primaries-b REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B]
                         [--reference-primaries-w REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W] [--display-primaries-r DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R]
                         [--display-primaries-g DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G] [--display-primaries-b DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B]
                         [--display-primaries-w DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W]

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
  --brightness BRIGHTNESS
                        brightness delta, -1.0 to 1.0, default = 0.0
  --contrast CONTRAST   contrast delta, 0.0 to 1.0, default = 0.0
  --hue HUE             hue angle delta, in degrees, default = 0.0
  --saturation SATURATION
                        saturation delta, -1.0 to 1.0, default = 0.0
  --phase-skew PHASE_SKEW
                        differential phase distortion, in degrees, default = 0.0
  --black-point BLACK_POINT
                        black point, in voltage units relative to blanking, default = 7.5/140.0 (7.5 IRE)
  --white-point WHITE_POINT
                        white point, in voltage units relative to blanking, default = 100.0/140.0 (100 IRE)
  --reference-primaries-r REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R
                        reference color primary R, in CIE xy chromaticity coordinates, default = [0.670, 0.330]
  --reference-primaries-g REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G
                        reference color primary G, in CIE xy chromaticity coordinates, default = [0.210, 0.710]
  --reference-primaries-b REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B
                        reference color primary B, in CIE xy chromaticity coordinates, default = [0.140, 0.080]
  --reference-primaries-w REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W
                        reference whitepoint, in CIE xy chromaticity coordinates, default = [0.313, 0.329]
  --display-primaries-r DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R
                        display color primary R, in CIE xy chromaticity coordinates, default = [0.640, 0.330]
  --display-primaries-g DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G
                        display color primary G, in CIE xy chromaticity coordinates, default = [0.300, 0.600]
  --display-primaries-b DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B
                        display color primary B, in CIE xy chromaticity coordinates, default = [0.150, 0.060]
  --display-primaries-w DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W
                        display whitepoint, in CIE xy chromaticity coordinates, default = [0.3127, 0.3290]

version 0.2.2
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
