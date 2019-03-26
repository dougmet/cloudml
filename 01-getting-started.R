library(here)
library(cloudml)

setwd(here("01-getting-started"))
getwd()

cloudml_train("iris.R")

# it's still setting up the environment after 30 minutes!

job_list()
## job_collect("cloudml_2019_03_26_204537248")
job_collect() # latest

# Let's do it again and see how long it takes
cloudml_train("iris.R")
# Took 9 minutes this time

job_collect("cloudml_2019_03_26_214758262")

model <- load_model_hdf5("runs/cloudml_2019_03_26_214758262/iris.hdf5")

model

get_weights(model)

### Evaluate and predict model

model %>% 
  evaluate(xIris$test, yIris$test)

model %>% 
  predict(xIris$test) %>%
  head()

model %>%
  predict_classes(xIris$test) 


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
