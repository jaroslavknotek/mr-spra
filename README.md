# Semester project 
This is supposed to be a semester project on multimedia retrieval subject. It demonstrates simple algorithms to extract data from spreadsheets and info graphics.

## Prerequisites
The project uses `frozen_east_text_detection.pb` from [here](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)


## Test and documentation
This project does not have any unit tests nor documentation.
The reason is an experiment with Jupyter Notebook as a "live" documentation. It describes what some function do 
and can execute it with a given input data. Notebooks can be found in `notebooks` directory.

## Notes

The project uses `logging` module configured to print everything(every level of importance) to console. 

## Data

Directory `input` contains test data.

## Future work

- Implement persistent storing
- Add result post-processing e.g. compute relations to other results
- Extend solution to non linear functions. It should work as follows
  - Extract layer with the function
  - Find center 
  - Approximate point of the function
  - Interpolate them and return it's approximate polynomial
- Replace OCR. It's very slow and does not perform well. 