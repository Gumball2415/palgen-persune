palgen_persune.py --skip-plot -e -o docs/example_palettes/2C02_default

palgen_persune.py --skip-plot -hue -15 -sat 0.65 --burst-saturation-disable -spg -gam 2.25 -rfc "NTSC (1953)" -rpw 0.3127 0.329 -e -o docs/example_palettes/savtool_replica

palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -cld -e -hue -3.75 -o docs/example_palettes/2C02-2C07_aps_ela_persune_neutral
palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -cld -e -hue 2.5 -o docs/example_palettes/2C02G_aps_ela_NTSC_persune_tink
palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -cld -e --burst-saturation-disable -o docs/example_palettes/2C02G_aps_ela_NTSC_persune_GVUSB2_NTSC_M_J -phs -5 -hue 12 -sat 0.8
palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -cld -e --burst-saturation-disable -o docs/example_palettes/2C02G_aps_ela_NTSC_persune_GVUSB2_NTSC_M -phs -5 -hue 12 -sat 0.8 -gai -6.5 -blp 6
palgen_persune.py --skip-plot -ppu "2C05-99" -sat 0.8 -cld -e -o docs/example_palettes/2C05-99_composite_forple

palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -phs -5.0 -blp 7.5 -e -o docs/example_palettes/2C02G_phs_aps_ela_NTSC
palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -phs -5.0 -blp 7.5 -rfc "NTSC (1953)" -e -o docs/example_palettes/2C02G_phs_aps_ela_NTSC-1953
palgen_persune.py --skip-plot -aps 5 -ela 0.01429 -phs -5.0 -rpr 0.618 0.350 -rpg 0.280 0.605 -rpb 0.152 0.063 -rpw 0.28314501 0.29711289 -e -o docs/example_palettes/2C02G_phs_aps_ela_NTSC-J

palgen_persune.py --skip-plot -ppu "2C07" -aps 5 -ela 0.01429 -phs -5.0 -blp 7.5 --delay-line-filter -cld -e -o docs/example_palettes/2C07_phs_aps_ela_PAL

palgen_persune.py --skip-plot -ppu "2C03" -rpr 0.622 0.338 -rpg 0.343 0.590 -rpb 0.153 0.059 -rpw 0.28314501 0.29711289 -e -o docs/example_palettes/2C03_DeMarsh_1980s_RGB