#read RPKM 
data_rpkm<-read.delim(file.choose())

#read annotation 
data_ann<-read.delim(file.choose())

data_ann_new<- data_ann[data_ann$BCR_ABL_status %in% c("negative","positive","normal"),]
data_ann_new$sign <- "0"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "negative"] <- "0"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "positive"] <- "1"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "normal"] <- "2"

#shouldBecomeOther<-!(data_ann$BCR_ABL_status %in% c("negative","positive","normal"))

require(xgboost)

#transpose the rpkm dataframe
data_rpkm <- as.data.frame(t(as.matrix((data_rpkm))))

#add a new column label into the rpkm dataset
data_rpkm$label <- 0

#save csv
write.csv(merged, 'merged.csv')

#split into train and validation sets
#training set: validation set = 70 : 30
set.seed(100)
train <- sample(nrow(merged), 0.7*nrow(merged), replace=FALSE)
TrainSet <- merged[train,]
ValidSet <- merged[-train,]

bst <- xgboost(data = merged[,] , label = merged$sign, max_depth=2, eta=1, nthread=2, nrounds=2, objective="binary:logistic")
xgb.importance()