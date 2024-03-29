# Pixel Art Extractor

## Description

A python script to extract the original pixel art from a PNG image, even if the pixel art has been scaled, translated, or rotated.

```
positional arguments:
  source_image          Filepath to the source image.

optional arguments:
  -h, --help            show this help message and exit
  -b, --border          Add a white border to the image.
  -s SCALE, --scale SCALE
                        Value to scale the final image up by.
```
## Installation and Usage

1. Install [Python 3](https://www.python.org/downloads/)
    * Make sure to add Python to PATH during installation
3. Install all the required packages using PIP
    * `pip install opencv-python`
    * `pip install argparse`
    * `pip install numpy`
    * `pip install matplotlib`
5. Download [pixel_art_extractor.py](https://raw.githubusercontent.com/dan-giddins/pixel-art-extractor/master/pixel_art_extractor.py) to a new folder
6. Open a terminal in the folder you have downloaded pixel_art_extractor.py to
    * In Windows, you can do this by opening file explorer and shift-right-clicking on the new folder you have just created, then clicking 'Open in Powershell'
7. In the command window, type `python pixel_art_extractor.py -h` and press enter to check everything is working
    * You should see a few lines of help text
8. Enter `python pixel_art_extractor.py` followed by a space and then the file path of the PNG source image
    * For example:
        * `python pixel_art_extractor.py C:\Path\To\File\picture.png`
    * If you have spaces in the filepath, you will need to add single quotes to the filepath argument:
        * `python pixel_art_extractor.py 'C:\Folder With Spaces\picture.png'`
1. A window of the source image will pop up, and you will be prompted to enter a pixel width value into the command line.
1. Some more windows will then pop up, showing you a debug view of the image, as well as the final pixelised image.
9. A new file should now be created called `pixel_art.png` in the same folder as `pixel_art_extractor.py`
    * WARNING: every time you run the script, it will overwrite `pixel_art.png`. If you want to keep the output, rename it and/or move it somewhere else
10. You can also use any combination of the following optional parameters:
    * `python pixel_art_extractor.py -b C:\Path\To\File\picture.png` if you would like the output to have a 1px wide border
    * `python pixel_art_extractor.py -s 5 C:\Path\To\File\picture.png` if you would like the output to be scaled up by a factor of 5 (this works for any positive integer)

## Example

Source image:

<img src="https://user-images.githubusercontent.com/16902799/112755782-2b7a2600-8fda-11eb-9803-ca76b0e38600.png" width="50%">

Hough line detection and pixel detection:

<img src="https://user-images.githubusercontent.com/16902799/112756106-7f393f00-8fdb-11eb-91c7-0ef425904d39.png" width="50%">

Output image (transparent, and with nice 1px border, then scaled up):

<img src="https://user-images.githubusercontent.com/16902799/112755805-3f258c80-8fda-11eb-9eb2-17660b18d7b2.png" width="50%">
