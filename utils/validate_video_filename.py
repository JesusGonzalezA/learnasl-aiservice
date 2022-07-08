ALLOWED_EXTENSIONS = ['MP4']

def validate_video_filename(filename):
	if filename == '':
		return False
	if not '.' in filename:
		return False

	extension = filename.rsplit('.', 1)[1]

	return extension.upper() in ALLOWED_EXTENSIONS