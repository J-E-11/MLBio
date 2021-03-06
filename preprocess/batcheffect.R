#author: Jinwan Huang
#Remove batch effect using limma library

data_rpkm<-read.delim(file.choose())
library(limma)
data_ann<-read.delim(file.choose())
batch <- data_ann$processing_date
data_rem_batch <- removeBatchEffect(data_rpkm, batch)

#Compare the results
library(compare)
comparison <- compare(data_rem_batch,data_rpkm,allowAll=TRUE)
comparison$tM

#Plot the difference between removing bacth effect and not
par(mfrow=c(1,2))
boxplot(as.data.frame(data_rpkm),main="Original")
boxplot(as.data.frame(data_rem_batch),main="Batch corrected")

#read annotation 
data_ann<-read.delim(file.choose())
data_ann_new<- data_ann[data_ann$BCR_ABL_status %in% c("negative","positive","normal"),]
data_ann_new$sign <- "0"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "negative"] <- "0"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "positive"] <- "1"
data_ann_new$sign[data_ann_new$BCR_ABL_status == "normal"] <- "2"
#transpose the rpkm dataframe
data_rem_batch <- as.data.frame(t(as.matrix((data_rem_batch))))

#Merge RPKM file and ANN file to get our metadata
rownames(data_ann_new) <- data_ann_new[,1]
data_ann_new[,1] <- NULL
merged <- merge(data_ann_new,data_rem_batch,by=0,all=TRUE)
merged[1:11] <-list(NULL)


write.csv(merged, 'merged_rm_be.csv')

