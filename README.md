# Description

[MNTox-app](https://leosotoj.com/pages/mntox/) is a demo app written in [Flask](https://flask.palletsprojects.com/) for the prediction of endocrine disruptive potency in environmental samples using tandem mass spectrometry. This [project page](http://leosotoj.com/pages/project-page) summarizes the main findings and provides further information with related links. 

# Installation

On Linux:

	git clone https://github.com/LeoSotoJ/mntoxapp
	cd mntoxapp
	virtualenv flaskenv
	source flaskenv/bin/activate
	pip install -r requirements.txt

To run Flask:

	flask --app app run
	 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)


# Structure and file contents

The main application is app.py. The views are located in the templates directory. The staticFiles contains the reference data used in the workflow. The client file is temporarily saved in the uploads directory and then deleted after processing.


	├── app.py
	├── Procfile
	├── requirements.txt
	├── staticFiles
	│   ├── example-DDA-features.csv
	│   ├── MassBank_net.msp
	│   ├── ref_net.graphml
	│   ├── tox.csv
	│   └── uploads/
	└── templates
	    ├── index.html
	    └── show_csv_data.html


