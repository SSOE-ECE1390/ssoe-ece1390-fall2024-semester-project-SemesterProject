[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tdy6BFPL)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15757940&assignment_repo_type=AssignmentRepo)
# SemesterProject
This is the template repository that will be used to hold the semester project


## Description

The real-time puzzle solver is a versatile system designed to interpret and solve puzzles directly from live video feed. Its capabilities include solving Sudoku puzzles in real time, and handling word search puzzles with a provided word bank. Additionally, it can identify symmetry or asymmetry between two images for "spot the difference" challenges and determine which of two playing cards has a higher value. This dynamic tool brings quick and accurate puzzle-solving to a variety of visual formats.

## Code Specifications

* General Inputs: Video Rate will be 24 FPS+, all puzzles will be on a sheet of paper with a white background. They should be held mostly still by the person holding the puzzle on the paper.  
* Sudoku:  
  * Inputs: Video(see above), with a solvable sudoku puzzle on a piece of paper that fills up 70% of the camera, while still being completely visible.   
  * Outputs: empty boxes on paper filled with numerical solutions to the puzzle   
* Word search:  
  * Inputs: Video(see above), a solvable word search on a piece on paper, user input of words to solve the puzzle  
  * Outputs: crossed out words on the puzzle  
* Spot the difference:   
* Inputs: Puzzle image in two file and mile and it must of the same size  
* Outputs: First puzzle image with circles of every difference it spots   
* Card game:   
  * Input: 2 or more playing cards in video, each fills \~30% of the camera side by side  
  * Output: Highlight or enlarge the card with the larger value

## Planned approach

This project has multiple elements, first is recognizing and processing the image. We will implement this with the OpenCV library for python. In the OpenCV library we plan to use a variety of algorithms to help isolate features, so that we can more easily recognize the puzzle, recognize the key features for solving the puzzle, and finally recognize where the answers need to go on the video feed to solve the puzzle. This would include background subtractor, or edge detection for isolating the paper. ORB for extracting the letter features, as well as OCR for recognizing and processing the characters to be used in the puzzle. Once we have found a variety of algorithms for solving puzzles (see below), on github. We will have to take the features that we have processed off of the puzzles and format them such that they can be utilized by the algorithms that we have found off of github. We may also have to modify the algorithms so that they output correctly onto the puzzle in the video feed, again using feature detection to find where these answers go on the puzzle. Finally, there may need to be some smoothing, in order to make sure that the user has a smooth experience.

## Time-line

1. Recognition of Paper  
2. Recognition of Letters and Numbers  
3. Sudoku Solving Algorithm  
4. Word Search Algorithm  
5. Taking a picture when the paper is correctly in frame  
6. Solving any two puzzles on a still image   
7. Solving any puzzle on a video  
8. Solving any two puzzles on video  
9. Solving every puzzle on a still image  
10. Solving every puzzle on a video  
11. Tracking the puzzle and where each of the solutions will go on said puzzle.

## Metrics of Success

The final goal for this project is to be able to solve all selected puzzles and display the solutions in real time. As we want to focus on the image processing for this project, the actual puzzle algorithms will be less of a priority.

## Pitfalls and alternative solutions

We foresee the spot the difference puzzle to be the most challenging of the puzzles to implement, as it is the only puzzle we’ve initially proposed that does not use words/text to solve and instead uses key features of a presented image to compare with key features of another presented image. As a backup, we can look into using different puzzles and potentially focusing solely on word/number puzzles. Some examples could include crossword, tetris, cryptogram, anagram, math solver, and brain teasers.

We also foresee potential issues when displaying the answers back on the sheet in real-time video, especially considering the precision and object tracking required for this task. As a potential backup here, we could instead print out a still image of the solution to the puzzle, rather than display it in the video in real-time.

* [OpenCV](https://opencv.org/)  
* [pytesseract](https://github.com/h/pytesseract)  
* [https://github.com/dhhruv/Sudoku-Solver](https://github.com/dhhruv/Sudoku-Solver)  
* [https://github.com/seancfong/Word-Search-Solver](https://github.com/seancfong/Word-Search-Solver)  
* [https://github.com/SSOE-ECE1390](https://github.com/SSOE-ECE1390)  
* [https://medium.com/@vinod.batra0311/solve-the-spot-the-differences-puzzle-with-computer-vision-2cb258fd2fc7](https://medium.com/@vinod.batra0311/solve-the-spot-the-differences-puzzle-with-computer-vision-2cb258fd2fc7) 

