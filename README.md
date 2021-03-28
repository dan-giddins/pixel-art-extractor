# pixel-art-extractor

A python script to extract the original pixel art from an image, even if the pixel art has been scaled, translated, or rotated.

```
positional arguments:
  source_image          Filepath to the source image.

optional arguments:
  -h, --help            show this help message and exit
  -b, --border          Add a white border to the image.
  -s SCALE, --scale SCALE
                        Value to scale the final image up by.
```
Source image:

<img src="https://user-images.githubusercontent.com/16902799/112755782-2b7a2600-8fda-11eb-9803-ca76b0e38600.png" width="50%">

Hough line detection and pixel detection:

<img src="https://user-images.githubusercontent.com/16902799/112756106-7f393f00-8fdb-11eb-91c7-0ef425904d39.png" width="50%">

Output image (transparent, and with nice 1px border, then scaled up):

<img src="https://user-images.githubusercontent.com/16902799/112755805-3f258c80-8fda-11eb-9eb2-17660b18d7b2.png" width="50%">
