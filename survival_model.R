##packages to install 
#install.packages('penalized')
#install.packages("survival)

library(penalized)
library(R.matlab)
library(survival)

# general format
#penalized (response, penalized, unpenalized, lambda1=0,
#           lambda2=0, positive = FALSE, data, fusedl=FALSE,
#           model = c("cox", "logistic", "linear", "poisson"),
#           startbeta, startgamma, steps =1, epsilon = 1e-10,
#           maxiter, standardize = FALSE, trace = TRUE)

#upload training features, survival time
train_data <- readMat('train_data.mat')
train_data <- as.data.frame(train_data[[1]])
train_surv <- readMat('train_surv.mat')
train_surv <- as.vector(train_surv[[1]])
attr(train_surv, "type") <- "right"
#train_event is for log classifier but we dont need that so i set it to 1
train_event <- rep(1, length(train_surv))

#same for testing data
test_data <- readMat('test_data.mat')
test_data <- as.data.frame(test_data[[1]])
test_surv <- readMat('test_surv.mat')
test_surv <- as.vector(test_surv[[1]])
attr(test_surv, "type") <- "right"
test_event <- rep(1, length(test_surv))

#parameters
lam1=1
lam2=1
#penalized function: Surv is a func from survival library
#penalized is the features that will be penalized with the lasso and ridge regression
#lam1 goes with lasso, lam2 goes with ridge
#i forget what positive does...in the manual
#data = data to train on
#model - theres options, we want the cox survival model
pen <- penalized(Surv(train_surv, train_event), penalized=train_data, unpenalized=~0, lambda1=lam1, lambda2=lam2, positive=FALSE, data=train_data, model=c("cox"))

#all coef of model, including non zero coef (betas)
#if you want coef nonzero then remove the "all"
all_coef <- coef(pen, "all")

#risk for each patient that the model trained on 
risk <- fitted(pen)


# predict using training data -- > output is breslow object
train_pred <- predict(pen, train_data, data=train_data)
plot(train_pred)
title("Training predictions")
#train_pred_mat: num patients x time steps -- each entry is probability of survival
train_pred_mat <- as.matrix(train_pred)

#predict using testing data --> output is breslow object
test_pred <- predict(pen, test_data, data=test_data)
plot(test_pred)
title("Testing predictions")
test_pred_mat <- as.matrix(test_pred)

#turn predictions mat into a vector with days until death
#ep=0.00001
#dead_thres=max(train_pred_mat[1:nrow(train_pred_mat),ncol(train_pred_mat)])+ep
#dead thres: at what survival probability is the patient "dead"
dead_thres=0.6
#time steps are the column names of the pred mat
time <- colnames(train_pred_mat)
#list of dead patients (empty -- at beg they are alive)
dead_pat <- c()
#initialize days of survival vector to all 0's 
s <- rep(0,nrow(train_pred_mat))
#loop through patients
for (i in 1:nrow(train_pred_mat)){
  #check if patient died yet -- i.e. are they in the dead list
  if (!is.element(i,dead_pat)){
    #if not dead, go through each time step
    for (j in 1:ncol(train_pred_mat)){
      prob=train_pred_mat[i,j]
      #find the first time step that drops below dead threshold probability
      if (prob < dead_thres){
        #enter their survival time into survival vec
        s[i] <- time[j]
        #add them to the dead pat list
        dead_pat <- c(dead_pat, i)
        break
      }
    }
  }
}

#calc accuracy for predictions
correct=0
#range of days to search
ep=365/2
train_pred_death <- as.numeric(s)
#loop through each patient
for (i in 1:length(train_surv)){
  #true days of survival
  true_td <- train_surv[i]
  #pred days of survival
  pred_td <- train_pred_death[i]
  #if pred days of survival is in interval: [true-ep, true+ep] 
  if (pred_td <= (true_td+ep) & pred_td >= (true_td-ep)){
    #correct prediction
    correct=correct+1
  }
}
#calc accuracy
train_accuracy = (correct / length(train_surv))*100

#mean squared error
tot_mse <- sum((train_pred_death-train_surv)**2)
#mean squared error given the slack of ep 
a<-abs(train_pred_death-train_surv)-ep
a[a<0]<-0
mse_slack <- sum(a**2)