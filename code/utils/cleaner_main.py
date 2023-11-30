
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

	print('Clean ' + file_python)

	# open file
	with open(file_python, "r") as f_in:
	    lines_in = f_in.readlines()

	# remove cell indices
	lines_in = [ line for i,line in enumerate(lines_in) if '# In[' not in line ]

	# remove comments
	lines_in = [ line for i,line in enumerate(lines_in) if line[0]!='#' ]

	# remove "in_ipynb()" function
	idx_start_fnc = next((i for i, x in enumerate(lines_in) if 'def in_ipynb' in x), None)
	if idx_start_fnc!=None:
	    idx_end_fnc = idx_start_fnc + next((i for i, x in enumerate(lines_in[idx_start_fnc+1:]) if x[:4] not in ['\n','    ']), None)  
	    lines_in = [ line for i,line in enumerate(lines_in) if i not in range(idx_start_fnc,idx_end_fnc+1) ]
	list_elements_to_remove = ['in_ipynb()', 'print(notebook_mode)']
	for elem in list_elements_to_remove:
	    lines_in = [ line for i,line in enumerate(lines_in) if elem not in line ]
	    
	# unindent "if notebook_mode==False" block
	idx_start_fnc = next((i for i, x in enumerate(lines_in) if 'if notebook_mode==False' in x), None)
	i