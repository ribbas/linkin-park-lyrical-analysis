# Makefile to ease trivial tasks for the project

VENV="$(shell find . -name ".*env")"
INVENV="$(shell which python | grep ${VENV})"
REQ="requirements.txt"

.PHONY: req-venv
# checks if virtual environment is activated and exits if it isn't 
req-venv:
ifeq (${INVENV}, "")
	$(error Virtual environment not activated)
endif


.PHONY: installenv
installenv:
	# install the virtual environment
	@test -d ${VENV} && virtualenv ${VENV} || virtualenv .venv


.PHONY: update
update: req-venv
	# update PIP requirements file
	@pip freeze > temp.txt
	@diff temp.txt jupyter-req.txt | grep "<" | cut -d " " -f 2 > ${REQ}
	@rm temp.txt


.PHONY: init
init: req-venv
	# upgrade PIP on virtual environment
	@pip install -U pip && pip install -r ${REQ}


.PHONY: notebook
notebook: req-venv
	# run Jupyter Notebook
	@jupyter nbextension enable --py --sys-prefix widgetsnbextension
	@jupyter notebook notebooks


.PHONY: convert-html
convert-html: req-venv
	# convert notebook to static HTML
	@jupyter nbconvert --to html notebooks/linkin-park-analysis.ipynb

.PHONY: clean-all
clean-all: clean reset


.PHONY: clean
clean:
	# clean out cache and temporary files
	@find . \( -name "*.pyc" -type f -o -name "__pycache__" -type d \) -delete


.PHONY: reset
reset:
	# remove distribution and raw data files
	@find . \( -path "./data/*" -o -path "./logs/*" \) -delete
	@find . \( -path "./data" -o -path "./logs" \) -empty -delete
