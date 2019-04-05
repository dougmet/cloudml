library(here)
library(cloudml)

# Need to be in right directory to submit job
# The here function anchors us to the project root
setwd(here("03-dogscats"))

# Submit a job
job <- cloudml_train("dogscats.R")

job_status(job)

job_gpu <- cloudml_train("dogscats.R", master_type = "standard_gpu")
job_status(job_gpu)

job_gpu_p100 <- cloudml_train("dogscats.R", master_type = "standard_p100")

job_list()

job_collect("cloudml_2019_04_04_215301128") # cpu run ~ 5.5  hours ~ $1
job_collect("cloudml_2019_04_04_224630201") # gpu run ~ 1.25 hours ~ $1
job_collect("cloudml_2019_04_05_093958377") # p100    ~ 0.33 hours ~ $0.60 WINNER!

view_run("cloudml_2019_04_04_224630201") 

model <- load_model_hdf5("runs/cloudml_2019_04_04_224630201/cats_and_dogs_small_2.h5")

# it's a normal model!
model

