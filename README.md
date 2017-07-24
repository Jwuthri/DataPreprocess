# Preprocess your csv files:

Step:
-----
	read the csv file, and transform into a pandas DataFrame
	regex the dataframe, format the string
	split the dataframe, remove useless columns (col_to_remove is a parameter), split into x, and y dataframes
	fill the dataframe, fill the NaN value using a clustering approach
	dummify the dataframe, transform categorics variables into dummies
	datify the dataframe, build new columns based on the datetime
	featurize the dataframe, make feature engineering like ploynomial featuring, derivate featuring, and transform featuring
	then scale the dataframe