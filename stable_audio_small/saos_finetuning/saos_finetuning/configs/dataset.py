def get_custom_metadata(info, audio):
	prompt = info['relpath'].replace("_", " ").replace("/", " ")
	return {"prompt": prompt}
