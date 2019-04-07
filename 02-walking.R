library(here)
library(cloudml)

# Need to be in right directory to submit job
# The here function anchors us to the project root
setwd(here("02-walking"))

# Submit a job
cloudml_train("walking.R")

# Collect a specific job
job_collect("cloudml_2019_03_26_214758262")
view_run()


# Load in the hdf5 model that we saved
model <- load_model_hdf5("runs/cloudml_2019_03_26_214758262/iris.hdf5")

# it's a normal model!
model
