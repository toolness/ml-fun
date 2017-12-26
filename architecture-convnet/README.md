This is an attempt at making a CNN that recognizes different
types of building architecture.

To use it, you'll first need to do a google image search for
different architecture types:

1. Do a search for "art deco building", click "show more results" at
   the bottom of the page, and then use "Save As..." in your browser
   to save the page. Move its images to `examples/art-deco`.

2. Do the same for "beaux arts building" and move its images
   to `examples/beaux-arts`.

Then run `train.py`.

Note that this project doesn't use the virtualenv used by the rest of
the projects in this repository. Instead, you'll probably want
to install Anaconda with keras.
