Install Ned Batchelder’s coverage module
http://nedbatchelder.com/code/coverage/
pip install coverage


Then, do this to get a basic coverage report on XNN:
nosetests --with-coverage --cover-package=xnn 

Incude the --cover-inclusive flag to get a more comprehensive report.





