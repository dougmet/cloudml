library(keras)
library(tidyverse)
library(rsample)
library(recipes)

## Create training and test sets

set.seed(367)

data_split <- initial_split(iris, strata = "Species", prop = 0.8)

fullData <- list(train = analysis(data_split), 
                 test = assessment(data_split))

# Preprocessing

iris_recipe <- recipe(Species ~ ., data = fullData$train) %>%
  step_dummy(Species, one_hot = TRUE, role = "outcome") %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep(training = fullData$train)

tidy(iris_recipe)

## Create x and y matrix

xIris <- map(fullData, ~ bake(object = iris_recipe,
                              new_data = .x,
                              all_predictors(),
                              composition = "matrix"))

yIris <- map(fullData, ~ bake(object = iris_recipe,
                              new_data = .x,
                              all_outcomes(),
                              composition = "matrix"))


############# Building models

model %>%
  keras_model_sequential() 
  layer_dense(units = 10, input_shape = 4) %>%
  layer_dense(units = 3, activation = 'softmax')


## Define compilation

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'accuracy'
)

## Train the model

history <- model %>% fit(xIris$train, 
                         yIris$train, 
                         epochs = 100, 
                         validation_data = list(xIris$test, yIris$test))

# This is for deployment scoring
export_savedmodel(model, "savedmodel")

# This is for getting the model back in
save_model_hdf5(model, "iris.hdf5")

