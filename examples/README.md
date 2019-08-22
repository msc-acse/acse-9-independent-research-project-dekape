This folder contains all the information required to reproduce the tomography and synthetic runs produced for this project.

The 'Runfiles' subfolder contains all SegyPreps and Fullwave3D runfiles and job-logs and should be descriptive of the physics and parameters applied to each run.

The models.xls spreadsheet provides a concise summary of the parameters and outcomes of each run.

The 'Models' subfolder contains the .sgy models of each tomography run. The outcomes of the synthetic runs are not included due to large storage requirements.

The 'project_overview' Jupyter Notebook visually present the evolution of the models produced for this project and show the updates made for subsequent model runs.

The 'parallel_workflow', 'sequential_workflow' and 'dd_workflow' notebooks present the proposed quality-controlled workflow applied to the timelapse methods for the final produced difference models PARDIFF25_18, SEQDIFF25_18 and DDDIFF25_18, respectively. It is recommended that the 'parallel_workflow' notebook is explored first as it contains more detailed signal analysis between the observed and predicted datasets. The other notebooks are a follow-up to the 'parallel_workflow' work and do not include as many presentations of the package. Although these notebooks make use of the true models and true datasets for the signal analysis, these are not provided in the repository as for intellectual property owned by Total E&P.

