## Example recreated from Deep Learning in R - Manning
## https://www.manning.com/books/deep-learning-with-r

library(keras)
library(cloudml)

bucket <- "dogscats_small"                       # for local
bucket <- "gs://rkeras-book174/dogscats_small"   # for mlengine

train_dir <- gs_data_dir_local(file.path(bucket, "train"))
validation_dir <- gs_data_dir_local(file.path(bucket, "validation"))

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# model$trainable_weights %>% length()

# Don't retrain the base
freeze_weights(conv_base)

# model$trainable_weights %>% length()

unfreeze_weights(conv_base, from = "block5_conv1")
model$trainable_weights %>% length()

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
test_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("dogscats-pretrain.hdf5")
