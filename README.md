# Data Mining and Machine Learning Coursework
by Antonio Gargaro, Nassim Abid and Dominic Calina

### Software Used
Built with Python 3.7 using the library python-weka-wrapper3. It allows you to use Weka from within Python by using the Javabridge library. This is required for using the Java Virtual Machine in which Weka processes get executed.
### Summary
As a group we decided to use the Python wrapper so that we had the ability to automate some processes like attribute selection, CSV randomisation and arff conversion. It also gave us the freedom to use the full data set; however, we still used smaller data sets to increase compile time when testing different algorithms.

### Directory 
```cmd
.
│   .gitignore                                  # Ignores files
│   config.py                                   # Change settings for weka in script
│   cw1-2018.pdf                                # CW File
│   F21DL_2018-2019-CW1Level10Rubric.pdf        # CW Rubric
│   helper.py                                   # Functions for preprocessing
│   HowTo.txt                                   # Cheat sheet to get started
│   README.md                                   # This file
│   requirements.txt                            # Dependencies for project
│   run.py                                      # Main python script
└───scripts
    │   wekaloader.py                           # Functions to use Weka
	│   bayes_networks.py                       # Bayes Network functionality
	│   clustering.py                           # Clustering functionality
	│   naiveBayes.py                           # Naive Bayes functionality
```
### What Files Do
- config.py
	- Specify directories for file creation
	- Specify the CSV directories e.g. fer2018.csv
	- Specify number of files and number of lines in randomised files e.g. 5 files and 7000 lines (cover bulk of dataset).
- helper.py
	- Methods for useful output of results in text files e.g. confusion matrix for part 4
	- Randomise the CSV files
	- Splits the pixels so that there are 2305 attributes
- run.py
	- Start the Weka wrapper
	- Make random CSV files if required
	- Run the Nearest Neighbour Algorithm
	- Select attributes using Ranker search method
- wekaloader.py
	- Convert files from CSV to ARFF
	- Change emotions from numeric to nominal
- bayes_networks.py
	- Runs bayesian network classifiers on data and outputs results
- clustering.py
	- Runs clustering algorithims on data and outputs results
- naiveBayes.py
	- Runs Naive Bayes classifiers on data and outputs
