
# Clean the main.py file after conversion from notebook.
# Any notebook code is removed from the main.py file.


import subprocess


def cleaner_main(filename):

	# file names
	file_notebook = filename + '.ipynb'
	file_python = filename + '.py'


	# convert notebook to python file
	print('Convert ' + file_notebook + ' to ' + file_python)
	subprocess.check_output('jupyter nbconvert --to script ' + str(file_notebook) , shell=True)

	pr