from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

# Set configuration for production
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['SECRET_KEY'] = os.urandom(24)  # Set a strong secret key

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 1 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
	if request.method == 'POST':
		f = request.files.get('file')

		# Ensure a file is selected
		if not f:
			flash('No file selected. Please choose a file to upload.', 'danger')
			return redirect(request.url)
		
		# Check if the file has the allowed extension
		if not allowed_file(f.filename):
			flash('Invalid file format. Only CSV files are allowed.', 'danger')
			return redirect(request.url)

		# Check if the file size is within the allowed limit (after reading the file content)
		# We use the `f.seek(0)` to ensure we reset the file pointer
		f.seek(0, os.SEEK_END)  # Move to the end to check the file size
		file_size = f.tell()    # Get the file size
		f.seek(0)               # Reset the pointer back to the beginning

		if file_size > app.config['MAX_CONTENT_LENGTH']:
			flash('File is too large. Please upload a file smaller than 1 MB.', 'danger')
			return redirect(request.url)

		# If the file passes both checks, save it
		data_filename = secure_filename(f.filename)
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
		f.save(file_path)
		
		# Store the file path in session
		session['uploaded_data_file_path'] = file_path
		
		# Flash success message
		flash('File uploaded successfully!', 'success')
		return redirect('/')
	
	return render_template("index.html")


# Validation for the Upload. Check if the file extension is allowed
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
		
# Validation for the Upload. Handle the 413 error (Request Upload Too Large)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
	flash('File is too large. Please upload a file smaller than 1 MB.', 'danger')
	return redirect(request.url)


@app.route('/load_example')
def load_example():
    # Simulate file upload by setting the example file path in session
    example_file_path = os.path.join('staticFiles', 'example-DDA-features.csv')
    session['uploaded_data_file_path'] = example_file_path

    # Flash success message
    flash('Example file loaded successfully!', 'success')

    # Redirect back to the main page to display the flash message
    return redirect('/')


@app.route('/download_example')
def download_example():
    # Path to the example file
    example_file_path = os.path.join('staticFiles', 'example-DDA-features.csv')
    
    # Check if the file exists
    if os.path.exists(example_file_path):
        return send_from_directory(
            directory='staticFiles',  # The directory containing the file
            path='example-DDA-features.csv',  # The file to download
            as_attachment=True  # This forces the browser to download the file
        )
    else:
        flash('Example file not found!', 'danger')  # Flash message if file not found
        return redirect('/')  # Redirect back to the homepagef

@app.route('/show_data')
def showData():

	# Uploaded File Path
	data_file_path = session.get('uploaded_data_file_path', None)
	
	# Read csv
	uploaded_df = pd.read_csv(data_file_path, encoding='utf-8-sig', header=None, index_col=0, sep=';')
	uploaded_df.columns = ['Raw spectra']

	result_df = uploaded_df

	# Processing the Upload CSV file 
    
    	## Create a list of spectrums
	import csv
	import numpy as np
	from matchms import Spectrum
	spectrums_features=[]
	with open(data_file_path, 'r', encoding='utf-8-sig') as file:
		reader = csv.reader(file, delimiter=';')
		for row in reader:
			if row[0] == 'id' or not row[0]:
				continue
			mz_intensities = row[1].split(' ')
			mz_values, intensity_values = zip(*[mz_intensity.split(':') for mz_intensity in mz_intensities])
			spectrum = Spectrum(mz=np.array(list(map(float, mz_values))),
					intensities=np.array(list(map(float, intensity_values))),
					metadata={'id': row[0], 'num_peaks': len(mz_values)})
			spectrums_features.append(spectrum)


    	## Mass Spectrums Pre-processing 
	import matchms.filtering as ms_filters
	from matchms import Spectrum
	
	def peak_processing(spectrum):
		spectrum = ms_filters.default_filters(spectrum)
		spectrum = ms_filters.normalize_intensities(spectrum)
		spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.05)
		return spectrum

	def peak_transformation(spectrum):
		spectrum = Spectrum(mz=spectrum.mz, intensities=np.sqrt(spectrum.intensities) , metadata=spectrum.metadata)
		return spectrum
	

	spectrums_features = [peak_processing(s) for s in spectrums_features]
	spectrums_features = [peak_transformation(s) for s in spectrums_features]
	spectrums_features=[s for s in spectrums_features if len(s.peaks)>=3]

	
    	# Saving the Pre-processed file

	from matchms.exporting import save_as_msp

	output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed', os.path.basename(data_file_path) + '_sample_features.msp')

	if os.path.exists(output_file):
	    os.remove(output_file)
	   
	    
	save_as_msp(spectrums_features, output_file)

	## Printing the pre-processed spectrums on a table

	import re

	file_path_s_msp = os.path.join(app.config['UPLOAD_FOLDER'],'processed/',os.path.basename(data_file_path)+'_sample_features.msp')
	
	with open(file_path_s_msp, 'r', encoding='utf-8') as file:
		content = file.read()

	paragraphs = content.split('\n\n')
	data = []
	
	for paragraph in paragraphs:
		lines = paragraph.split('\n')
		first_line = lines[0]
		id_number = re.findall(r'\d+', first_line)  # Find the first number in the first line
	    
		if id_number:  # If a number is found, use it as the index
			data.append((int(id_number[0]), " ".join(lines)))  # Store the id and paragraph content
		else:
			continue

	paragraphs_df = pd.DataFrame(data, columns=['Index', 'Processed spectra: features with minimum 3 peaks of 0.05 relative intensity after processing'])
	paragraphs_df.set_index('Index', inplace=True)
	result_df = pd.concat([result_df, paragraphs_df], axis=1)

	## Network Analysis and collecting the script results for each spectrum

	import settings 
	import networkx as nx
	from matchms.importing import load_from_msp
	
	spectrums_features = list(load_from_msp(file_path_s_msp)) # Import sample spectra, ref net spectra, and tox table
	spectrums_net = list(load_from_msp(os.path.join('staticFiles', 'MassBank_net.msp')))
	network = nx.read_graphml(os.path.join('staticFiles', 'ref_net.graphml'))
	tox_dict = pd.read_csv(os.path.join('staticFiles', 'tox.csv')).set_index('inchikey').to_dict(orient='index')

	from matchms.similarity import CosineGreedy
	from matchms import calculate_scores
	from datetime import datetime, date, time
	import io
	import sys
	
	alerts_feature_id=[]
	features_id=[]
	alert_paragraphs = []
	result_paragraphs = []

	for f in spectrums_features:	# Loop through the spectrums_features
		features_id.append(np.int64(f.get('id')))
		# Capture the print output for each feature
		output = io.StringIO()  # Create a string buffer to capture the prints
		sys.stdout = output  # Redirect print statements to the buffer

		print('ID: ', f.get('id'), end=" ")
		connected_active = 0
		connected_inactive = 0

		try:
			similarity_measure = CosineGreedy(tolerance=0.1)
			scores_f = calculate_scores([f], spectrums_net, similarity_measure, is_symmetric=False)
			scores_array_f = scores_f.scores.to_array()
		except Exception as e:
			print(f"Error processing spectrum {f}: {e}. Calculate_scores returns empty as index if there is not any matched peaks.", end=" ")
			result_paragraphs.append(f"ID {f.get('id')}:\n Calculate_scores returns empty as index because there is not any matched peak with the reference network")
			continue

		filtered_indexes = np.where(np.logical_and(scores_array_f[0]['CosineGreedy_matches'] >= 3, scores_array_f[0]['CosineGreedy_score'] >= 0.2))

		for index in filtered_indexes[0]:
			inchikey = spectrums_net[index].get('inchikey')
			activity = tox_dict[inchikey]['NR.AR']
			print(inchikey, end=" ")
			print(activity, end=" ")
			if activity == 1:
				connected_active += 1 
			if activity == 0:
				connected_inactive += 1

		print('\nconnected_active', connected_active, end=" ")
		print('connected_inactive', connected_inactive, end=" ")

		if connected_active >= connected_inactive and connected_active != 0:
			print("\n<span style='color:red;'>Alert: active connected nodes equal or greater than inactive nodes</span>", end=" ")
			alerts_feature_id.append(np.int64(f.get('id')))

		print(end="")  # For a blank line to separate the output for each feature

		# Store the captured output and reset the print redirection
		result_paragraphs.append(output.getvalue())  # Store the output in the list
		sys.stdout = sys.__stdout__  # Reset the print redirection back to normal

	# To display line breaks in the table
	result_paragraphs = [s.replace("\n", "<br>") for s in result_paragraphs]

	# Create a column with all network connections

	print(len(features_id))
	print(len(result_paragraphs))

	output_df = pd.DataFrame({
		'Feature ID': features_id,  # Use the collected feature IDs
        'Connections with the reference network: InChIKeys and NR.AR activities': result_paragraphs  # Use the collected paragraphs
	})
	
	output_df.set_index('Feature ID', inplace=True)
	
	result_df = pd.concat([result_df, output_df], axis=1)
	
	
	# Create a column with Alerts only

	#alerts_df = output_df.loc[output_df.index.isin(alerts_feature_id)]
	#alerts_df.columns.values[0] = 'Alerts of the suspected NR.AR activity of the feature'
	#result_df = pd.concat([result_df, alerts_df], axis=1)


	
	# Highlight the rows where the Feature ID is in alerts_feature_id by adding a style (OVERWRITES BOOTSTRAP)
	def highlight_alerts(row):
	    return ['background-color: #FFFACD' if row.name in alerts_feature_id else '' for _ in row]
	# Apply the style to the DataFrame
	result_df = result_df.style.apply(highlight_alerts, axis=1)


	# Converting the df to an HTML table with applied styles
	uploaded_df_html = result_df.to_html(classes='table table-sm table-bordered table-hover', escape=False)


	return render_template('show_csv_data.html', data_var=uploaded_df_html)




if __name__ == '__main__':
	app.run()
