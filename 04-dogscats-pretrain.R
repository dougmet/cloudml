library(here)
library(cloudml)

# Need to be in right directory to submit job
# The here function anchors us to the project root
setwd(here("04-dogscats-pretrain"))

# Submit a job
cloudml_train("dogscats-pretrain.R", master_type = "standard_gpu")
cloudml_train("dogscats-pretrain.R", master_type = "standard_p100")

job_list()

job_collect("cloudml_2019_04_05_093043851") # gpu standard
job_collect("cloudml_2019_04_05_094300772") # p100
view_run("cloudml_2019_04_05_094300772") # p100

# For everything I've measured P100 is not faster than K80

model <- load_model_hdf5("runs/cloudml_2019_04_05_093043851/dogscats-pretrain.hdf5")

# it's a normal model!
model

