######################### BLACK FRIDAY #################################

####################### PROBLEM STATEMENT #############################
# Predict the purchase price of house (Regression)

#Clear R enviroment
rm(list = ls())

#Setting working directory
setwd("D:/Data Science/R CODE/Black Friday")

#install and load the package
x <- c("xlsx", "ggplot2", "plyr", "data.table", "DT", "gmodels", "dummies", "h2o", "caTools", "DMwR", "usdm", "corrplot", "Metrics")
install.packages(x) #Installs all the packages
sapply(x, require, character.only = TRUE)

#Loading data in R using data.table
train <- fread("train.csv", stringsAsFactors = TRUE)
test <- fread("test.csv", stringsAsFactors = TRUE)

#combine data set
#creating Purchase Var for test data and assigning mean
test[,Purchase:= mean(train$Purchase)]
combin <- rbindlist(list(train, test))

############# Exploratory data analysis #################
#Analysing Distribution
#Age vs Gender
ggplot(combin, aes(Age, fill = Gender)) + geom_bar() + 
       labs(title = "Age Distribution")

#Age vs City_Category
ggplot(combin, aes(Age, fill = City_Category)) + geom_bar() +
       labs(title = "City Distribution")
      
#analyzing gender variable
prop.table(table(combin$Gender))

#Age Variable
prop.table(table(combin$Age))

#City Category Variable
prop.table(table(combin$City_Category))

#Stay in Current Years Variable
prop.table(table(combin$Stay_In_Current_City_Years))

#unique values in ID variables
length(unique(combin$Product_ID)) #3677 Unique values
length(unique(combin$User_ID)) #5891 Unique values
#Can be used for feature engineering

#missing values
colSums(is.na(combin)) # Too many missing values in Product_category 2 and 3
#Can be used for feature engineering


#Making modifications in variables

#changing column level 
combin[,Stay_In_Current_City_Years:= ifelse(Stay_In_Current_City_Years== "4+", 4, Stay_In_Current_City_Years)]

#recoding age groups
combin[,Age:= ifelse(Age == "0-17",0,Age)]
combin[,Age:= ifelse(Age == "18-25",1,Age)]
combin[,Age:= ifelse(Age == "26-35",2,Age)]
combin[,Age:= ifelse(Age == "36-45",3,Age)]
combin[,Age:= ifelse(Age == "46-50",4,Age)]
combin[,Age:= ifelse(Age == "51-55",5,Age)]
combin[,Age:= ifelse(Age == "55+",6,Age)]

#convert Gender into numeric
combin[,Gender:= as.numeric(Gender)]

######## Feature Engineering ##########
#User Count by User ID
combin[, User_Count := .N, by = User_ID]

#Product Count by Product ID
combin[, Product_Count := .N, by = Product_ID]

#create a new variable for missing values
combin[,Product_Category_2_NA := ifelse(sapply(Product_Category_2,is.na)==TRUE,1,0)]
combin[,Product_Category_3_NA := ifelse(sapply(combin$Product_Category_3, is.na) == TRUE,1,0)]

#impute missing values
combin[,Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "999",  Product_Category_2)]
combin[,Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "999",  Product_Category_3)]

#Mean Purchase of Product
combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]

#Mean Purchase of User
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]

#one hot encoding of City_Category variable
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_")

#check classes of all variables
sapply(combin, class)

#converting Product Category 2 & 3
combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)

#Divide into train and test
c.train <- combin[1:nrow(train),]
c.test <- combin[-(1:nrow(train)),]

#Removing noise from product_category_1 
c.train <- c.train[c.train$Product_Category_1 <= 18,] # Test data has 18 levels, Train data has 20 levels

#Checking Correlation
#set the upper triangle to be zero and then remove any rows that have values over particular value
#temp <- cor(c.train[,3:19])
#temp[!lower.tri(temp)] <- 0
#new.train <- c.train[, !apply(temp, 2, function(x) any(x > 0.8))]

# Plotting correlation matrix
corrplot(cor(c.train[,3:19]), order= "hclust", method = "square")

install.packages("ggcorrplot")
require("ggcorrplot")
ggcorrplot(cor(c.train[,3:19]), hc.order = TRUE, type = "upper",
           lab = TRUE, title = "Correlation plot using ggcorrplot")


vif(c.train[,3:19])


#Creating data sample for cross validation
set.seed(123)
s.train <- c.train[sample(nrow(c.train),10000, replace = FALSE),]
s.test <- c.train[sample(nrow(c.train),2500, replace = FALSE),]

s.train <- s.train[, c(3:13,15:19,14)]
s.test <- s.test[,c(3:13,15:19,14)]
s.test2 <- s.test[,1:(ncol(s.train)-1)]
  

#launch the H2O cluster
localh2o <- h2o.init(nthreads = -1)

#data to h2o cluster
train.h2o <- as.h2o(s.train)
test.h2o <- as.h2o(s.test2)

#check column index number
colnames(train.h2o)

#dependent variable (Purchase)
y.dep <- ncol(s.train)

#independent variables (dropping ID variables)
x.indep <- 1:(ncol(s.train)-1)

#Multiple Regression in H2O
#GLM algorithm in H2O can be used for all types of regression such as lasso, ridge, logistic, linear etc. 
#only needs to modify the family parameter accordingly
#logistic regression,write family = "binomial".
regression.model <- h2o.glm( y = y.dep, x = x.indep,
                             training_frame = train.h2o,
                             family = "gaussian")

h2o.performance(regression.model)

#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))

#calculating MAPE
regr.eval(s.test[,17], predict.reg, stats = 'mape') 

#Random Forest
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.indep, 
                                    training_frame = train.h2o,
                                    ntrees = 1000, mtries = 1,
                                    max_depth = 1, seed = 1122))

h2o.performance(rforest.model)
h2o.r2(rforest.model)

#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(
  predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o))
  )

#calculating MAPE
regr.eval(s.test[,17], predict.rforest, stats = 'mape') 


#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep,
                       training_frame = train.h2o,
                       ntrees = 1000, max_depth = 1,
                       learn_rate = 0.05, seed = 1122)
)

h2o.performance(gbm.model)
#Displays R-Squared value calculated by model
h2o.r2(gbm.model)

#making prediction and writing submission file
system.time(
  predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
)

#calculating MAPE
regr.eval(s.test[,17], predict.gbm, stats = 'mape') 

#deep learning models
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(5,5),
                                      activation = "Rectifier",
                                      seed = 1122
  )
)

h2o.performance(dlearning.model)
h2o.r2(dlearning.model)

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#calculating MAPE
regr.eval(s.test[,17], predict.dl2, stats = 'mape') 

