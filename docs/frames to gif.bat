ffmpeg -framerate 2 -i "QAM sequence %%03d.png" -filter_complex "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" "QAM sequence.gif"
ffmpeg -framerate 2 -i "waveform sequence %%03d.png" -filter_complex "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" "waveform sequence.gif"
ffmpeg -framerate 2 -i "palette sequence %%03d.png" -filter_complex "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" "palette sequence.gif"
pause
