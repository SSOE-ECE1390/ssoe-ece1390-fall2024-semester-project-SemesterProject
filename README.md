[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tdy6BFPL)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15975195&assignment_repo_type=AssignmentRepo)
# SemesterProject
This is the template repository that will be used to hold the semester project

## ECE 1390 White Page
### Description
For our proposed project, we intend to implement a facial filter using two images the user will upload. The first image may contain the user, while the second image may contain any face/shape that the user would like to place over their own using a filter, outputting an edited version of the first image as the result. Additionally, we will be implementing a Bokeh effect on the output image. Typically, the Bokeh effect blurs the background of an image with circular points of focus, but we intend to use the face/shape provided in the second user inputted image to re-shape these points of focus.


### Code Specifications
We plan to use python to implement our facial filter and Bokeh effect. We will likely be using multiple image processing libraries provided in python, including OpenCV, SciPy, Scikit-Image, and others.

### Planned Approach
We plan to divide up each required method so that each person implements 1-2 methods to satisfy the requirements. We will separate into two groups, one which focuses on the filtering algorithms, and one which focuses on the facial recognition algorithm. The filtering algorithms will include edge detection to determine the shape of the cartoon face/shape, the boke convolution blur based on that edge detected shape, and general image enhancement (such as auto-contrast/saturation). The facial recognition team will likely work with ML, though this will partially depend on the topics covered in lectures.

### Timeline
We plan to complete most of the basic algorithms by the end of October, and then refine/add more functionality afterwards. The facial recognition algorithm will be completed last as additional knowledge is picked up in later lectures.


We will also hold weekly meetings to communicate on progress and collaborate.


### Metrics of Success
Our metrics of success will be determined in a few ways. We want to prioritize each of the requirements of the assignment, i.e. image loading/saving, image enhancement, image filtering, edge detection, segmentation, and object recognition. In addition to this, we want to prioritize our program’s ability to recognize facial features, fully separate the background from the subject of the image (a person’s head and body), full replacement of the first image subject’s head, and full functionality of the Bokeh effect. 

### Pitfalls/Alternative Solutions
There are several edge cases that could cause issues with our filter. For example, faces at sharp angles, or if a face is partially cut off. For these specific cases with the facial recognition algorithm, we will have to tweak the algorithm/training to try to reduce these cases, or we could also apply an algorithm to check if input pictures are valid (e.g. that a face is detected). When there are multiple faces in the image, we plan to replace all of them.
