# load required packages
library(Rcpp)
library(ggplot2)
library(caret)
library(xgboost)
library(randomForest)
library(gbm)
library(MASS)
library(Matrix)
library(boot)

# read training and testing dataset
TRAIN <- read.csv("C:/Users/Evelyn Stamey/Documents/GitHub/Santander/Santander_train.csv")
TEST <- read.csv("C:/Users/Evelyn Stamey/Documents/GitHub/Santander/Santander_test.csv")

# remove id column
TRAIN$ID <- NULL
TEST_ID <- TEST$ID
TEST$ID <- NULL

# remove response column (TARGET)
TRAIN_Y <- TRAIN$TARGET
TRAIN$TARGET <- NULL

# find and remove vectors that are linear combinations of other vectors
LINCOMB <- findLinearCombos(TRAIN)
TRAIN <- TRAIN[, -LINCOMB$remove]
TEST <- TEST[, -LINCOMB$remove]

# find and remove vectors with near zero variance
NZV <- nearZeroVar(TRAIN, saveMetrics=TRUE)
TRAIN <- TRAIN[,-which(NZV[1:nrow(NZV),]$nzv==TRUE)]
TEST <- TEST[,-which(NZV[1:nrow(NZV),]$nzv==TRUE)]

# find and remove vectors that are highly corrolated to other vectors
HIGHCOR <- findCorrelation(cor(TRAIN[,1:ncol(TRAIN)]), cutoff = .95, verbose = FALSE)
TRAIN <- TRAIN[,-HIGHCOR]
TEST <- TEST[,-HIGHCOR]

# re-attach response vector to TRAIN
TRAIN$TARGET <- TRAIN_Y

# convert TRAIN dataframe into a design matrix
TRAIN_MAT <- sparse.model.matrix(TARGET ~ ., data = TRAIN)
D_TRAIN <- xgb.DMatrix(data = TRAIN_MAT, label = TRAIN_Y)
WATCHLIST <- list(TRAIN_MAT = D_TRAIN)

# define xgb.train parameters
PARAM <- list(objective        = "binary:logistic", 
              booster          = "gbtree",
              eval_metric      = "auc",
              eta              = 0.01,
              max_depth        = 6,
              subsample        = 0.68,
              colsample_bytree = 0.68,
              gamma            = 0,
              alpha            = 0
)

# set seed
set.seed(6352) 

# run cross-validation
CV1 <- xgb.cv(params      = PARAM, 
              data        = D_TRAIN, 
              nrounds     = 1000, 
              nfold       = 5,
              verbose     = 2
)
MAX_AUC_INDEX <- which.max(CV1[, test.auc.mean])

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = D_TRAIN, 
                   nrounds     = MAX_AUC_INDEX, 
                   verbose     = 2,
                   watchlist   = WATCHLIST
)

# attach a predictions vector to the test dataset
TEST$TARGET <- -1

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("TARGET")
TEST_MAT <- sparse.model.matrix(TARGET ~ ., data = TEST)
PRED <- predict(MODEL, TEST_MAT)

# create submission file
SUBMIT <- data.frame(ImageId = c(1:length(PRED)), Label = PRED)
write.csv(SUBMIT, "C:/Users/Evelyn Annette/Documents/Team Evelyn/Santander Kaggle/submit.csv", row.names = FALSE)










