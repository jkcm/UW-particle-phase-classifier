# UW-particle-phase-classifier: the University of Washington Ice-Liquid Discriminator (UWILD) random forest particle phase classification scheme

This software is currently developed at the University of Washington to classify individual cloud particles as seen by the 2DS probe, using a random forest approach. V1.0 will be released alongside an academic publication (DOI to be added upon creation), which describes the development and results of this scheme. 



## Current Developers (July 2020)
- **Rachel Atlas (ratlas@uw.edu)
- **Joe Finlon (jfinlon@uw.edu)
- **Ian Hsiao (ianjjhsiao@gmail.com)
- **Jeremy Lu (jlu43@uw.edu)
- **Hans Mohrmann (jkcm@uw.edu)

## Dependencies
This project was primarily developed using Python 3.7.9. 
Required packages:

- matplotlib 
- netCDF4 
- numpy 
- pandas 
- scipy 
- sklearn 
- xarray 






Inputs to the classifier originate in the [UIOPS probe data processing software](https://github.com/joefinlon/UIOPS). Any work referencing this project should additionally and separately reference the UIOPS project (DOI:10.5281/zenodo.3667054). 

## Project layout
- /python_scripts/: all data processing code for generating datasets, training model, etc.
- /postprocessing_visualization/: all code for generating plots manuscript figures
- /notebooks/: misc notebooks used for project development. Not cleaned up, included for reference
- /model_data/: contains pickled trained model.
