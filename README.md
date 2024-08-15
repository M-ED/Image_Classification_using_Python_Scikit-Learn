
# Image Classification using Scikit-Learn and OpenCV

- This project demonstrates a simple image classification pipeline using Python, Scikit-Learn, and OpenCV.
- The goal is to classify images into two categories: empty and non_empty.
- The dataset is prepared by loading images, resizing them, and then training a Support Vector Classifier (SVC) to predict the category of test images.






## Prerequisites
- Python 3.9.0 or higher 
- OpenCV
- NumPy
- Scikit-Learn

## Installation

1. Clone the repository:

```bash
  git clone https://github.com/M-ED/Image_Classification_using_Python_Scikit-Learn
```

2. Create virtual environment using following commands:
```bash
  conda create -n projects_CV python==3.9.0
  conda activate projects_CV
```

3. Install the necessary libraries in requirements file
```bash
   pip install -r requirements.txt
```

4. Run the script
```bash
  python main.py
```


## Features

Following are the key features:
- clf-data/: Directory containing subfolders `empty` and `non_empty` with images for classification.
- util.py: Contains helper functions like `check_directory`.
- main.py: The main script that handles image loading, preprocessing, model training and evaluation.



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

- OpenCV: [https://opencv.org/](https://opencv.org/)
- Scikit-Learn: [hhttps://scikit-learn.org/stable//](https://scikit-learn.org/stable/)



## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Author

- [@mohtadia_naqvi](https://github.com/M-ED)

