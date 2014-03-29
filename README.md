/**
 * If you intend to use this code for research purposes, please cite:
 * E. Sariyanidi, H. Gunes, M. Gokmen, A. Cavallaro 
 * 	  'Local Zernike Moment Representation for Facial Affect Recognition'
 * 	  BMVC'13
 * 
 * This code is licensed under CreativeCommons Non-Commercial license 3.0
 * 	  http://creativecommons.org/licenses/by-nc/3.0/deed.en_GB
**/

Guide to compile and run QLZM feature extractor.
Please contact e.sariyanidi[at]qmul.ac.uk for any questions and comments, bug reports etc.


Libraries needed: OpenCV (core, highgui and imgproc). Tested with 2.3.1, should work with 2.3+

For Linux:
----------
Make sure that OpenCV libraries are accessible to the executable file (i.e. they are installed to the system e.g. /usr/local/lib ...)
Then you may follow these steps to compile and run:
	1) Extract QLZM.zip and change directory to QLZM (cd QLZM)
	2) Compile the project with Makefile:
		$ make
	3) Run executable
		$ ./QLZM path-to-image.png 1

The QLZM executable takes two arguments. The first is a string to the image file. The second controls whether features will be saved to a file (with the argument 1) or just dumped to the terminal (with the argument 0). 



For Windows:
------------
You should be able to run code using QTCreator IDE (or qmake if you know how to use).
For QTCreator, you can use the project file provided (QLZM.pro).

You need to modify the QLZM.pro file:
1) Add path to OpenCV headers by modifying INCLUDEPATH
2) Add path to OpenCV libraries -- you may do this from the Project Tree in QTCreator (by right-click to QLZM project then "Add Library").

Then Build the project and you can run it either from the IDE or from command line using the compiled QLZM executable.

The QLZM executable takes two arguments. The first is a string to the image file. The second controls whether features will be saved to a file (with the argument 1) or just dumped to the terminal (with the argument 0). 


