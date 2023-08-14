cd ..
palgen-persune.py --skip-plot -p -w -r
palgen-persune.py --skip-plot -e -r -t docs/demo_screenshots/smb.bin
palgen-persune.py -h
echo 
pause
call "make palettes.bat"
cd docs
call "frames to gif.bat"