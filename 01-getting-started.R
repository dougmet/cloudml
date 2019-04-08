library(here)
library(cloudml)

# Need to be in right directory to submit job
# The here function anchors us to the project root
setwd(here("01-getting-started"))

# Submit a job
cloudml_train("iris.R")

job_list()
# it's still setting up the environment after 30 minutes!
job_collect()

# Let's do it again and see how long it takes
cloudml_train("iris.R")
# Took 9 minutes this time

# Collect a specific job
job_collect("cloudml_2019_03_26_214758262")

# Look at the history
view_run("cloudml_2019_03_26_214758262")

# Load in the hdf5 model that we saved
model <- load_model_hdf5("runs/cloudml_2019_03_26_214758262/iris.hdf5")

# it's a normal model!
model


# Evaluate and predict model ----------------------------------------------


model %>% 
  evaluate(xIris$test, yIris$test)

model %>% 
  predict(xIris$test) %>%
  head()

model %>%
  predict_classes(xIris$test) 


# Deployment model --------------------------------------------------------

library(tfdeploy)

predict_savedmodel(xIris$test, "runs/cloudml_2019_03_26_214758262/savedmodel")


# start session
sess <- tensorflow::tf$Session()

# preload an existing model into a TensorFlow session
graph <- tfdeploy::load_savedmodel(sess, "runs/cloudml_2019_03_26_214758262/savedmodel")

tfdeploy::predict_savedmodel(
  c(-0.1, 1, -1, -1),
  graph,
  sess = sess
)

sess$close()
