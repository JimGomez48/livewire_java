James Gomez
CS 269 Final Project: A "Live-wire" Implementation in Java using JavaCV and OpenCV
Winter 2014
Professor Terzopoulos

/************************************************************************************
This implementation is based on the paper "Interactive live-wire boundary extraction"
by William A. Barrett and Eric N. Mortensen.
************************************************************************************/


PREFERRED PLATFORM/ENVIRONMENT
==================================
- Linux (Must have gui-based desktop environment)
- Windows
NOTE: Not tested in Mac OS X whatsoever


REQUIREMENTS
==================================
- A working 32-bit installation of OpenCV. Refer to
  <http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html> 
  for complete installation instructions.
- A working 32-bit installation of the java runtime environment,
  version 7 (jre7) or later.

NOTE: Both OpenCV and Java MUST be 32-bit for this application to work. 
      Ensure environment variables point to 32-bit builds
      (x86/vc10 for OpenCV)


INSTRUCTIONS FOR RUNNING PROGRAM
==================================
- Unzip the project
- Ensure the res/ folder and livewire.jar file are in the same directory
- In a terminal(or command prompt) cd to the project directory and
  type "java -jar livewire_java.jar <path to image file>"
  Images have been provided in the res/ folder for convenience.


IN-APP INSTRUCTIONS
==================================
- Left-click near an edge to generate a starting seed point.
- Drag the cursor around the image to adjust the boundary.
- To cool current boundary, left-click again near a desired edge.
- To clear current boundary, double-click RIGHT mouse button.
- To close off boundary, overlap free end with current boundary tail
  and left-click. The app will detect boundary closure and stop
  tracing. It will then display the extracted boundary and image
  segment.
- Double-click the LEFT mouse button over the live-wire app window to 
  save the current segment to disk, or double-click the RIGHT mouse 
  button to clear the current boundary.
- Press any key at any time to exit the app.
