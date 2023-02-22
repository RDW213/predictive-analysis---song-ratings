library(dplyr)
library(tidyverse)
library(caret)
library(xgboost)
#library(vtreat)
library(qdapTools) 

set.seed(1025)

songs <- read.csv('analysisData.csv')
scoringData <- read.csv('scoringData.csv')

#scr <- data.matrix(scoringData)

#song <- songs[,-c(1,2,3,5,9,17)]
#scordata <- scoringData[,-c(1,2,3,5,9,17)]

song <- songs[,-c(1,3)]
scordata <- scoringData[,-c(1,3)]

#feature engineering - genre

song$genre <- gsub('\\[','',song$genre)
song$genre <- gsub('\\]','',song$genre)
song$genre <- gsub("\\'",'',song$genre)

song$genre <- strsplit(song$genre,',')
song <- cbind(song, mtabulate(song$genre))

scordata$genre <- gsub('\\[','',scordata$genre)
scordata$genre <- gsub('\\]','',scordata$genre)
scordata$genre <- gsub("\\'",'',scordata$genre)

scordata$genre <- strsplit(scordata$genre,',')
scordata <- cbind(scordata, mtabulate(scordata$genre))

shared_g <- intersect(names(song), names(scordata))

scordata <- select(scordata,all_of(shared_g))
song <- select(song,c('rating',all_of(shared_g)))

#feature engineering - performer

song$performer <- strsplit(song$performer,' ')
song <- cbind(song, mtabulate(song$performer))

scordata$performer <- strsplit(scordata$performer,' ')
scordata <- cbind(scordata, mtabulate(scordata$performer))

shared_p <- intersect(names(song), names(scordata))

scordata <- select(scordata,all_of(shared_p))
song <- select(song,c('rating',all_of(shared_p)))

#feature engineering - track_explicit

scordata$track_explicit <- as.numeric(scordata$track_explicit)
song$track_explicit <- as.numeric(song$track_explicit)


# split into training and testing dataset
song <- song[,-c(2,3)]
scordata <- scordata[,-c(1,2)]

parts <- createDataPartition(song$rating, p = .8, list = F)
train <- song[parts, ]
test <- song[-parts, ]


#define predictor and response variables in training set
train_x <- data.matrix(train[, -1])
train_y <- train[,'rating']

#define predictor and response variables in testing set
test_x <- data.matrix(test[, -1])
test_y <- test[, 'rating']


# testing

ames_xgb <- xgb.cv(
  data = as.matrix(train_x),label = train_y,nrounds = 6000,objective = "reg:squarederror",
  early_stopping_rounds = 50,nfold = 10,params = list(eta = 0.1,max_depth = 3,
                                                      min_child_weight = 3,subsample = 0.8,colsample_bytree = 1.0),verbose = 0) 

min(ames_xgb$evaluation_log$test_rmse_mean)


# hyperparameter grid
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = c(3,6), 
  min_child_weight = c(1,3,5),
  subsample = c(0.2,0.5,0.8), 
  colsample_bytree = c(0.2,0.5,0.8),
  gamma = c(0, 1, 10),
  lambda = c(0, 1e-2, 0.1, 1, 100),
  alpha = c(0, 1e-2, 0.1, 1, 100),
  rmse = 0,          # a place to dump RMSE results
  trees = 0          # a place to dump required number of trees
)





# grid search
for(i in seq_len(nrow(hyper_grid))) 
  { m <- xgb.cv(
    data = train_x,
    label = train_y,
    nrounds = 6000,
    objective = "reg:squarederror",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i], 
      lambda = hyper_grid$lambda[i], 
      alpha = hyper_grid$alpha[i]
    ) 
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration}

# results
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()


# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5,
  gamma=0,
  lambda=1e+00,
  alpha=0)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = as.matrix(train_x),
  label = train_y,
  nrounds = 3147,
  objective = "reg:squarederror",
  verbose = 0)



pred <- predict(xgb.fit.final,newdata=as.matrix(scordata))

RMSE <- sqrt(mean((test_y - pred)^2))


  
submissionFile = data.frame(id = scoringData$id, rating = pred)
write.csv(submissionFile, 'sample_submission_XGBoost.csv',row.names = F)
