[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tdy6BFPL)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=16029232&assignment_repo_type=AssignmentRepo)
# SemesterProject
**Description**<br>
Our project aims to build off the idea of a face swap application, where facial recognition is applied to images to detect facial expressions on people and replace their face with the corresponding face emoji.  The program will detect whether a person is smiling, frowning, crying, etc., and swap their face with the respective emoji counterpart that has the same facial expression.  Add-on features of the program will include detecting hair, accessories, and clothing on the person and replacing them with their emoji counterparts as well.<br><br>

**Code Specification**<br>
Inputs:<br>
- Upper-torso only color input images 
- Library of emojis to swap with 

Outputs:<br>
- Image with features/face swapped with their emoji counterparts<br><br>


**Planned Approach**<br>
First, we plan to find a way to detect faces in photos, and from there be able to compare and swap them with an emoji by detecting the borders of the face and the shape and borders of facial features that will be used to determine the input facial expression.  If possible, we will also detect someone’s hair and facial hair, and superimpose their own hair, or an emoji-esque version of it, onto the emoji that represents their expression.<br><br>

**Time-line**<br>
- Milestone 1: Detecting an entire face in photos 
- Milestone 2: Compare facial expression and swap with appropriate emoji 
- Milestone 3: Detect more features on a face like hair, facial hair and swap with a more specific emoji 
- Milestone 4: Detect clothing and swapping with emoji<br><br>
 
**Metrics of Success**<br>
We will feed input images with expressions that are close to specific emojis in the emoji library – for example, a picture of someone with an angry expression, a happy expression, etc. We will run these images through our program and judge whether the output image uses the emoji with the most closely related expression, and whether the emoji is placed in the correct position directly over the subject’s head. If the output has the expected emoji and covers the subject’s head within a reasonable margin of error, that image will be a success.<br><br>
 
**Pitfalls and Alternative Solutions**<br>
We may find that the last few steps of our project could be too difficult.  At the bare minimum, we would like to be able to replace the person’s head with the correct emoji expression. Our success with adding various expressions and getting these expressions to be detected reliably will determine how much we focus on detecting hair, facial hair, and other specific facial features to be replaced with their emoji counterparts. 
