##########################################
#Loading packages
##########################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(OpenImageR)) install.packages("OpenImageR", 
                                          repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", 
                                           repos = "http://cran.us.r-project.org")
if(!require(stringi)) install.packages("stringi", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

#Intro
#Reference:
#https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721


##########################################
#Data Wragling
##########################################


#We can get the data from Kaggle - basicshapes
#"https://www.kaggle.com/cactus3/basicshapes"

########
##unzip("basicshapes.zip")
##unzip("shapes.zip")

#shapes
shapes = c("circles","squares","triangles")

####read images to create a dataset of images
fig_dataset = lapply(shapes, function(x){
  file_names = paste(x,"\\",dir(x),sep="")
  read_files = lapply(file_names, function(y){
    fig = readImage(y)
    shape = x
    return(list(img = fig,shape = shape))
  })
})

class(fig_dataset)

###dataset is list of 3 list, one for each shape. 
###It is better to have a unique list for all shapes, so will use unlist 
fig_dataset = unlist(fig_dataset, recursive = F) 

class(fig_dataset)
length(fig_dataset)

###each list is a list with a img object that represents our images 
###shape field is the shape of the images 
names(fig_dataset[[1]])

###we can check that object of img is a array
class(fig_dataset[[1]]$img)

###each dimension of the matrix represents a matrix of RGB scale
class(fig_dataset[[1]]$img[,,1])

###############
####Shapes
###########

###circle 
image(fig_dataset[[1]]$img[,,1])

###square
image(fig_dataset[[101]]$img[,,1])

###triangle
image(fig_dataset[[201]]$img[,,1])

###Augmentation
#usar image augmentation -> aumentar o número de amostras através da manipulação das já existentes
#https://www.rdocumentation.org/packages/OpenImageR/versions/1.1.5/topics/Augmentation
#https://hazyresearch.github.io/snorkel/blog/tanda.html


###transformar a imagem RGB em GRAY Scale
img = rgb_2gray(img)


#original image
image(rgb_2gray(fig_dataset[[210]]$img))

###flip
image(rgb_2gray(flipImage(fig_dataset[[210]]$img, mode = "vertical")))

###rotate
image(rgb_2gray(rotateFixed(fig_dataset[[210]]$img, 90)))


###HOG
#The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision
# and image processing for the purpose of object detection. 


#http://www.vlfeat.org/overview/hog.html


########################################################
#Train and test set
########################################################

####createdatapartion (img directly)
####because our dataset is a list and not a data.frame,
####we will first create a vector with the same size of our list 
#### Then we will use it as a index to split our data

vec_img = 1:300
set.seed(1)
test_index = createDataPartition(vec_img,times = 1, p = 0.2, list = F)

test_set = fig_dataset[test_index]
train_set = fig_dataset[-test_index]

rm(vec_img)

###função para transformar em escala de cinza, augmentation e HOG (obtém features da figura)
GRAY_AUG_HOG = function(picture){
  pict = rgb_2gray(picture)
  img_flip_vert = HOG(flipImage(pict, mode = "vertical"))
  img_flip_hor = HOG(flipImage(pict, mode = "horizontal"))
  img_rot_90 = HOG(rotateFixed(pict, 90))
  img_rot_180 = HOG(rotateFixed(pict, 180))
  img_rot_270 = HOG(rotateFixed(pict, 270))
  img = HOG(pict)
  df = rbind(img,img_flip_vert,img_flip_hor,img_rot_90,img_rot_180,img_rot_270)
  return(df)
}

####Train dataset increased by Augmentation and transformed by HOG
train_df = lapply(train_set, function(x){
  fig = GRAY_AUG_HOG(x$img)
  data.frame(fig)%>% mutate(shape = x$shape)
})
train_df = plyr::ldply(train_df)

train_df


####create test set
test_df = lapply(test_set, function(x){
  fig = HOG(x$img)
  data.frame(t(fig),shape = x$shape)
})
test_df = plyr::ldply(test_df)
test_df

###################################################
#train models
###################################################

####models to be trained
models <- c("lda",  "naive_bayes",  "svmLinear", "qda", 
            "knn", "kknn", "loclda",
            "rf", "wsrf", 
            "avNNet", "mlp", "monmlp","gbm",
            "svmRadial", "svmRadialCost", "svmRadialSigma")

####train models
fit = lapply(models, function(x){
  print(x)
  fit = train_df %>% train(shape~., data = ., method = x)
})

###accuracy
acc = sapply(fit, function(x){
  max(x$results$Accuracy, na.rm = T)
})
names(acc) = models
acc


###Tuning the models
###we can see how models where tuned, and check if they need further improvement using ggplot
ggplot(fit[[16]])


###model loclda
fit_loclda=  train_df %>% train(shape~., data = ., method = "loclda",
                                tuneGrid = data.frame(k = seq(325,370,5)))

###model svmRadialSigma
fit_svmRadialSigma =
  train_df %>%
  train(shape~., data = .,method = "svmRadialSigma",
        tuneGrid = expand.grid(sigma = seq(0.015,0.06,0.005), C = c(1,2)))
fit_svmRadialSigma$results
ggplot(fit_svmRadialSigma)
max(fit_svmRadialSigma$results$Accuracy)

ggplot(fit_loclda)

#############################################
#Ensemble
#############################################

model_acc =data.frame(model = models, acc = acc)
model_ok = model_acc %>% filter(acc > 0.85)
model_ok
acc
pred = predict(fit[acc>0.85])


pred = data.frame(matrix(unlist(pred), nrow = nrow(train_df)))
names(pred) = model_ok$model

pred_cl = rowSums(pred == "circles")
pred_sq = rowSums(pred == "squares")
pred_tr = rowSums(pred == "triangles")



preds = data.frame(pred, circle = pred_cl, square = pred_sq, triangle = pred_tr) %>%
  mutate(pred = ifelse(circle>square & circle>triangle,"circles",
                       ifelse(square>circle & square>triangle,"squares", 
                              ifelse(triangle>circle &triangle>square, "triangles", 
                                     as.character(wsrf)))))

confusionMatrix(as.factor(preds$pred), as.factor(train_df$shape))

###necessário identificar qual modelo melhor se adapta para casos de igualdade
####################################
######Accuracy test set
###################################

pred_test = predict(fit[acc>0.85], newdata = test_df)
pred_test = data.frame(matrix(unlist(pred_test), nrow = nrow(test_df)))
names(pred_test) = model_ok$model

pred_cl_test = rowSums(pred_test == "circles")
pred_sq_test = rowSums(pred_test == "squares")
pred_tr_test = rowSums(pred_test == "triangles")

preds_test = data.frame(pred_test, circle = pred_cl_test, square = pred_sq_test, triangle = pred_tr_test) %>%
  mutate(pred = ifelse(circle>square & circle>triangle,"circles",
                       ifelse(square>circle & square>triangle,"squares", 
                              ifelse(triangle>circle &triangle>square, "triangles", 
                                     as.character(wsrf)))))

preds_test$pred

########models
CMpredloclda = confusionMatrix(as.factor(preds_test$loclda), as.factor(test_df$shape))$overall[[1]]
CMpredloclda

CMpredsvmRadialSigma = 
  confusionMatrix(as.factor(preds_test$svmRadialSigma), as.factor(test_df$shape))$overall[[1]]
CMpredsvmRadialSigma

########ensemble model
CMpred = confusionMatrix(as.factor(preds_test$pred), as.factor(test_df$shape))
CMpred$overall[[1]]