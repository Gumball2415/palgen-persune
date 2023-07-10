cd ..
call "make palettes.bat"
palgen-persune.py --skip-plot -p -w -r
palgen-persune.py --skip-plot -e -r
palgen-persune.py -h
echo 
pause
cd docs
call "frames to gif.bat"