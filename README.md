# Palette Generator

yet another NES palette generator, in Python

<img src="docs/palette preview.png">
<img src="docs/waveform phase.gif">
<img src="docs/QAM phase.gif">
<img src="docs/palette preview emphasis.gif">

something to note: there _is_ no one true NES palette, but this generator
can pretty much approach colors that looks good enough.
feel free to adjust to taste!

## Requirements

See `requirements.txt` for more details.

### This script requires:

- `colour-science == 0.4.6`
	- for linear light, and color adaptation functions. 
- `matplotlib == 3.9.3`
	- for graphs and `colour-science` CIE 1931 colorimetry diagrams.

## Usage
```
usage: palgen_persune.py [-h] [-d] [--skip-plot] [-o OUTPUT]
                         [-f {.pal uint8,.pal double,.pal Jasc,.gpl,.png,.txt HTML hex,.txt MediaWiki,.h uint8_t}]
                         [-e] [-t TEST_IMAGE] [-r RENDER_IMG] [-w] [-p]
                         [-n {scale,scale clip negative}]
                         [-ppu {2C02,2C03,2C04-0000,2C04-0001,2C04-0002,2C04-0003,2C04-0004,2C05-99,2C07}]
                         [-c {darken,desaturate}] [-bri BRIGHTNESS]
                         [-con CONTRAST] [-hue HUE] [-sat SATURATION]
                         [-blp BLACK_POINT] [-whp WHITE_POINT] [-gai GAIN]
                         [-gam GAMMA] [--delay-line-filter]
                         [-axs {None,CXA2025AS_JP,CXA2025AS_US,bisqwit_NTSC_1953}]
                         [-bsd] [-spg] [-phs PHASE_SKEW]
                         [-aps ANTIEMPHASIS_PHASE_SKEW]
                         [-ela EMPHASIS_LUMA_ATTENUATION]
                         [-rfc REFERENCE_COLORSPACE] [-dsc DISPLAY_COLORSPACE]
                         [-cat CHROMATIC_ADAPTATION_TRANSFORM] [-ict]
                         [-oetf OPTO_ELECTRONIC] [-eotf ELECTRO_OPTIC]
                         [--opto-electronic-disable] [--electro-optic-disable]
                         [-cld]
                         [-rpr REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R]
                         [-rpg REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G]
                         [-rpb REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B]
                         [-rpw REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W]
                         [-dpr DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R]
                         [-dpg DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G]
                         [-dpb DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B]
                         [-dpw DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W]

yet another NES palette generator

options:
  -h, --help            show this help message and exit
  -d, --debug           debug messages
  --skip-plot           skips showing the palette plot
  -o OUTPUT, --output OUTPUT
                        file output path
  -f {.pal uint8,.pal double,.pal Jasc,.gpl,.png,.txt HTML hex,.txt MediaWiki,.h uint8_t}, --file-format {.pal uint8,.pal double,.pal Jasc,.gpl,.png,.txt HTML hex,.txt MediaWiki,.h uint8_t}
                        file output format. default = ".pal uint8"
  -e, --emphasis        include emphasis entries
  -t TEST_IMAGE, --test-image TEST_IMAGE
                        use 256x240 uint16 raw binary PPU frame buffer for
                        palette proofreading
  -r RENDER_IMG, --render-img RENDER_IMG
                        render views and diagrams as images in docs folder
                        with the provided file extension.
  -w, --waveforms       view composite waveforms
  -p, --phase-QAM       view QAM demodulation
  -n {scale,scale clip negative}, --normalize {scale,scale clip negative}
                        normalize all colors within gamut by scaling values
  -ppu {2C02,2C03,2C04-0000,2C04-0001,2C04-0002,2C04-0003,2C04-0004,2C05-99,2C07}, --ppu {2C02,2C03,2C04-0000,2C04-0001,2C04-0002,2C04-0003,2C04-0004,2C05-99,2C07}
                        PPU chip used for generating colors. default = 2C02
  -c {darken,desaturate}, --clip {darken,desaturate}
                        clips out-of-gamut RGB colors
  -bri BRIGHTNESS, --brightness BRIGHTNESS
                        luma brightness delta in IRE units, default = 0.0
  -con CONTRAST, --contrast CONTRAST
                        luma contrast factor, default = 1.0
  -hue HUE, --hue HUE   chroma hue angle delta, in degrees, default = 0.0
  -sat SATURATION, --saturation SATURATION
                        chroma saturation factor, default = 1.0
  -blp BLACK_POINT, --black-point BLACK_POINT
                        black point, in IRE units, default = 0 IRE
  -whp WHITE_POINT, --white-point WHITE_POINT
                        white point, in IRE units, default = level $20
  -gai GAIN, --gain GAIN
                        gain adjustment to signal before decoding, in IRE
                        units, default = 0.0
  -gam GAMMA, --gamma GAMMA
                        if defined, will apply a simple OETF gamma transfer
                        function instead, where the EOTF function is assumed
                        to be gamma 2.2.
  --delay-line-filter   use 1D delay line comb filter decoding instead of
                        single-line decoding
  -axs {None,CXA2025AS_JP,CXA2025AS_US,bisqwit_NTSC_1953}, --axis-shift {None,CXA2025AS_JP,CXA2025AS_US,bisqwit_NTSC_1953}
                        axis adjustment for R-Y and G-Y like Sony CXA2025AS,
                        default = None
  -bsd, --burst-saturation-disable
                        disable using colorburst amplitude as saturation
                        reference
  -spg, --sinusoidal-peak-generation
                        generate sine waves in composite encoding instead of
                        square waves
  -phs PHASE_SKEW, --phase-skew PHASE_SKEW
                        differential phase distortion for composite PPUs, in
                        degrees, default = 0.0
  -aps ANTIEMPHASIS_PHASE_SKEW, --antiemphasis-phase-skew ANTIEMPHASIS_PHASE_SKEW
                        additonal phase distortion on colors $x2/$x6/$xA for
                        composite PPUs, in degrees, default = 0.0
  -ela EMPHASIS_LUMA_ATTENUATION, --emphasis-luma-attenuation EMPHASIS_LUMA_ATTENUATION
                        additonal luma brightness on colors $x4/$x8/$xC for
                        composite PPUs, in voltage units, default = 0.0
  -rfc REFERENCE_COLORSPACE, --reference-colorspace REFERENCE_COLORSPACE
                        use colour.RGB_COLOURSPACES reference colorspace,
                        default = "SMPTE C"
  -dsc DISPLAY_COLORSPACE, --display-colorspace DISPLAY_COLORSPACE
                        use colour.RGB_COLOURSPACES display colorspace,
                        default = "sRGB"
  -cat CHROMATIC_ADAPTATION_TRANSFORM, --chromatic-adaptation-transform CHROMATIC_ADAPTATION_TRANSFORM
                        chromatic adaptation transform method, default = None
  -ict, --inverse-chromatic-transform
                        invert direction of chromatic adaptation transform
                        method (from display to reference colorspace)
  -oetf OPTO_ELECTRONIC, --opto-electronic OPTO_ELECTRONIC
                        applies "colour.models" color component transform
                        function to use as opto-electronic transform function
                        override
  -eotf ELECTRO_OPTIC, --electro-optic ELECTRO_OPTIC
                        applies "colour.models" color component transform
                        function to use as electro-optic transform function
                        override
  --opto-electronic-disable
                        disable converting linear light to linear signal
  --electro-optic-disable
                        disable converting linear signal to linear light
  -cld, --colorimetry-disable
                        disable all colorimetry functions
  -rpr REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R, --reference-primaries-r REFERENCE_PRIMARIES_R REFERENCE_PRIMARIES_R
                        set custom reference color primary R, in CIE xy
                        chromaticity coordinates
  -rpg REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G, --reference-primaries-g REFERENCE_PRIMARIES_G REFERENCE_PRIMARIES_G
                        set custom reference color primary G, in CIE xy
                        chromaticity coordinates
  -rpb REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B, --reference-primaries-b REFERENCE_PRIMARIES_B REFERENCE_PRIMARIES_B
                        set custom reference color primary B, in CIE xy
                        chromaticity coordinates
  -rpw REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W, --reference-primaries-w REFERENCE_PRIMARIES_W REFERENCE_PRIMARIES_W
                        set custom reference whitepoint, in CIE xy
                        chromaticity coordinates
  -dpr DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R, --display-primaries-r DISPLAY_PRIMARIES_R DISPLAY_PRIMARIES_R
                        set custom display color primary R, in CIE xy
                        chromaticity coordinates
  -dpg DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G, --display-primaries-g DISPLAY_PRIMARIES_G DISPLAY_PRIMARIES_G
                        set custom display color primary G, in CIE xy
                        chromaticity coordinates
  -dpb DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B, --display-primaries-b DISPLAY_PRIMARIES_B DISPLAY_PRIMARIES_B
                        set custom display color primary B, in CIE xy
                        chromaticity coordinates
  -dpw DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W, --display-primaries-w DISPLAY_PRIMARIES_W DISPLAY_PRIMARIES_W
                        set custom display whitepoint, in CIE xy chromaticity
                        coordinates

version 0.16.0
```

## License

This work is licensed under the MIT-0 license.

Copyright (C) Persune 2024.

## Credits

Special thanks to:
- NewRisingSun
- lidnariq
- PinoBatch
- jekuthiel
- _aitchFactor

This would have not been possible without their help!
