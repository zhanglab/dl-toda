library("ggplot2")
library("ggforce")
library("PupillometryR") #for making half violin plot
library("fitdistrplus")
library("EnvStats")
library("stats")
library("tidyr")
library("ggpubr")

args = commandArgs(trailingOnly = TRUE)
input_dir = args[1]
#a <- read.table(file=file.path(paste(input_dir, "updated-testing-set-all-out-ncbi-cutoff-0.0-species-confidence-scores.tsv", sep="/")),
#               sep="\t")
a <- read.table(file=file.path(paste(input_dir, "updated-testing-set-all-out.tsv", sep="/")),
               sep="\t")
print(a)
##### 1. ###########
#plot both correct and incorrect in one plot
#ggplot(data=a, mapping=aes(x=V4,fill=V2)) +
#  geom_histogram(alpha = 0.5, bins = 50) +
#  labs(x = "Prediction Scores", y = "Count",
#       title = "Distribution of DL-TODA prediction scores") +
#  theme_bw()

#-plot the scores of only incorrect predictions together with the fitted curve
#ggplot(data=a[a$V2=='incorrect',], mapping=aes(x=V4,fill=V2)) +
#  geom_histogram(alpha = 0.5, bins = 50, fill="grey") +
#  labs(x = "Prediction Scores", y = "Count",
#       title = "Distribution of DL-TODA prediction scores - incorrect predictions") +
#  theme_bw()

#plot the scores of only correct predictions
#ggplot(data=a[a$V2=='correct',], mapping=aes(x=V4,fill=V2)) +
#  geom_histogram(alpha = 0.5, bins = 50, fill="black") +
#  labs(x = "Prediction Scores", y = "Count",
#       title = "Distribution of DL-TODA prediction scores - correct predictions") +
#  theme_bw()


##### 2. ###########
#Fit gamma distributions to the scores of incorrect predictions
f<-fitdist(a[a$V3=='incorrect',]$V4,distr="gamma")
#Fitted result:
print(summary(f))
#Fitting of the distribution ' gamma ' by maximum likelihood 
#Parameters : 
#  estimate  Std. Error
#shape 4.036873 0.001255371
#rate  8.384575 0.002776566
#Loglikelihood:  1864960   AIC:  -3729916   BIC:  -3729887 
#Correlation matrix:
#  shape      rate
#shape 1.0000000 0.9390747
#rate  0.9390747 1.0000000

#-prepare data for plotting the fitted gamma distribution
x <- seq(0,1,length=21)
print(x)
gamma_dist <- data.frame(cbind(x, dgamma(x,f$estimate[1],f$estimate[2])))
hist(a[a$V3=='incorrect',4],ylim=c(0,2),freq=FALSE,xlab="Prediction Scores",
     main="Distribution of DL-TODA prediction scores - incorrect predictions")
lines(gamma_dist$x,gamma_dist$V2,col="red")

##### 3. ###########
# the analysis below gives the following estimations:
#--With a cutoff of 0.54, we expect to filter out ~65% of the incorrect reads 
#  and retain 95% of the correct reads
#--With a cutoff of 0.64, we expect to filter out 75% of the incorrect reads 
#  and retain 92% of the correct reads.
#--With a cutoff of 0.72, we expect to filter out 85% of the incorrect reads
#--With a cutoff of 0.8, we expect to filter out 90% of the incorrect reads
#--With a cutoff of 0.94, we expect to filter out 95% of the incorrect reads

quantile(a[a$V3=='incorrect',]$V4, probs=c(0,0.05,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1))
e_incorrect<-eqgamma(a[a$V3=='incorrect',]$V4, p=0.90,
                     ci=TRUE, ci.type="upper")

#-prepare data for plotting the eqgamma fitted distribution of incorrect
x <- seq(0,1,length=21)
gamma_incorrect <- data.frame(cbind(x, dgamma(x,shape=e_incorrect$parameters[1],
                                              scale=e_incorrect$parameters[2])))
hist(a[a$V3=='incorrect',4],ylim=c(0,2),freq=FALSE,xlab="Prediction Scores",
     main="Distribution of DL-TODA prediction scores - incorrect predictions")
lines(gamma_incorrect$x,gamma_incorrect$V2,col="red")


quantile(a[a$V3=='correct',]$V4, probs=c(0,0.01,0.05,0.06,0.07,0.08,0.1,0.5,0.9,0.95,1))
e_correct<-eqgamma(a[a$V3=='correct',]$V4, p=0.1,
                   ci=TRUE, ci.type="lower")

#-prepare data for plotting the eqgamma fitted distribution of correct
x <- seq(0,1,length=21)
gamma_correct <- data.frame(cbind(x, dgamma(x,shape=e_correct$parameters[1],
                                              scale=e_correct$parameters[2])))
hist(a[a$V3=='correct',4],ylim=c(0,2),freq=FALSE,xlab="Prediction Scores",
     main="Distribution of DL-TODA prediction scores - correct predictions")
lines(gamma_correct$x,gamma_correct$V2,col="red")

##### 4. ###########
## The section below is used when a summary result (e.g. 'precision_mtx_0.94.tsv')
## is not already parsed with a given threshold
species<-unique(a$V1)
pmtx<-data.frame(matrix(ncol=2,nrow=0))
pmtx_0.54<-data.frame(matrix(ncol=2,nrow=0))
pmtx_0.64<-data.frame(matrix(ncol=2,nrow=0))
pmtx_0.7<-data.frame(matrix(ncol=2,nrow=0))
##--a cutoff of 0.8 correspond to filtering out 90% of the false positives based on eqgamma
pmtx_0.8<-data.frame(matrix(ncol=2,nrow=0))
pmtx_0.9<-data.frame(matrix(ncol=2,nrow=0))
##--a cutoff of 0.94 correspond to filtering out 95% of the false positives based on eqgamma
pmtx_0.94<-data.frame(matrix(ncol=2,nrow=0))
for(s in species){
  ## no cutoff
  #tp<-nrow(a[a$V1==s & a$V2=='correct',])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect',])
  tp<-nrow(a[a$V1==s & a$V2==s,])
  fp<-nrow(a[a$V1!=s & a$V2==s,])
  precision<-tp/(tp+fp)
  pmtx[nrow(pmtx)+1,]<-c(s,precision)
  ## cutoff 0.54
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.54,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.54,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.54,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.54,])
  precision<-tp/(tp+fp)
  pmtx_0.54[nrow(pmtx_0.54)+1,]<-c(s,precision)
  ## cutoff 0.64
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.64,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.64,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.64,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.64,])
  precision<-tp/(tp+fp)
  pmtx_0.64[nrow(pmtx_0.64)+1,]<-c(s,precision)
  ## cutoff 0.7
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.7,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.7,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.7,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.7,])
  precision<-tp/(tp+fp)
  pmtx_0.7[nrow(pmtx_0.7)+1,]<-c(s,precision)
  ## cutoff 0.8
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.8,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.8,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.8,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.8,])
  precision<-tp/(tp+fp)
  pmtx_0.8[nrow(pmtx_0.8)+1,]<-c(s,precision)
  ## cutoff 0.9
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.9,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.9,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.9,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.9,])
  precision<-tp/(tp+fp)
  pmtx_0.9[nrow(pmtx_0.9)+1,]<-c(s,precision)
  ## cutoff 0.94
  #tp<-nrow(a[a$V1==s & a$V2=='correct' & a$V4>0.94,])
  #fp<-nrow(a[a$V1==s & a$V2=='incorrect' & a$V4>0.94,])
  tp<-nrow(a[a$V1==s & a$V2==s & a$V4>0.94,])
  fp<-nrow(a[a$V1!=s & a$V2==s & a$V4>0.94,])
  precision<-tp/(tp+fp)
  pmtx_0.94[nrow(pmtx_0.94)+1,]<-c(s,precision)
}
write.table(pmtx,file="precision_mtx.tsv",sep="\t",quote=FALSE,
           col.names = F, row.names = F)
write.table(pmtx_0.54,file="precision_mtx_0.54.tsv",sep="\t",quote=FALSE,
            col.names = F, row.names = F)
write.table(pmtx_0.64,file="precision_mtx_0.64.tsv",sep="\t",quote=FALSE,
           col.names = F, row.names = F)
write.table(pmtx_0.7,file="precision_mtx_0.7.tsv",sep="\t",quote=FALSE,
           col.names = F, row.names = F)
write.table(pmtx_0.8,file="precision_mtx_0.8.tsv",sep="\t",quote=FALSE,
           col.names = F, row.names = F)
write.table(pmtx_0.9,file="precision_mtx_0.9.tsv",sep="\t",quote=FALSE,
           col.names = F, row.names = F)
write.table(pmtx_0.94,file="precision_mtx_0.94.tsv",sep="\t",quote=FALSE,
            col.names = F, row.names = F)

# ## Make a summary of precision by different cutoffs
precision_df<-data.frame(matrix(ncol=6,nrow=639))
colnames(precision_df)<-c("species_gt",
                          "c_0.54","c_0.64","c_0.7","c_0.8","c_0.94")
precision_df$species_gt<-pmtx_0.54[,1]
precision_df$c_0.54<-as.numeric(pmtx_0.54[,2])
precision_df$c_0.64<-as.numeric(pmtx_0.64[,2])
precision_df$c_0.7<-as.numeric(pmtx_0.7[,2])
precision_df$c_0.8<-as.numeric(pmtx_0.8[,2])
precision_df$c_0.9<-as.numeric(pmtx_0.9[,2])
precision_df$c_0.94<-as.numeric(pmtx_0.94[,2])

## The section below is used when a summary result (e.g. 'precision_mtx_0.94.tsv')
## is already present
fs<-c('precision_mtx_0.54.tsv','precision_mtx_0.64.tsv',
      'precision_mtx_0.7.tsv','precision_mtx_0.8.tsv',
      'precision_mtx_0.94.tsv')
precision_df<-data.frame(matrix(ncol=6,nrow=639))
colnames(precision_df)<-c("species_gt",
                          "c_0.54","c_0.64","c_0.7","c_0.8","c_0.94")


for(i in 1:length(fs)){
  f<-fs[i]
  print(f)
  pmtx<-read.table(file=file.path(paste(input_dir, f, sep="/")), sep="\t", comment.char='')
  precision_df[,i+1]<-as.numeric(pmtx[,2])
}
precision_df$species_gt<-pmtx[,1]


precision_df_long<-pivot_longer(precision_df,
                                cols=c("c_0.54","c_0.64","c_0.7","c_0.8","c_0.94"),
                                names_to = "Pred_Score_cutoff",
                                values_to = "Precision")

#--Plot Figure 5A
plot_1 <- ggplot(precision_df_long, aes(x=Pred_Score_cutoff, y=Precision)) +
  geom_boxplot(outlier.colour="black", outlier.shape=NA,
               outlier.size=1, outlier.alpha=0.1,notch=FALSE, width=0.5) +
  geom_jitter(width = 0.2, size=0.5, alpha=0.3) +
  scale_x_discrete(labels=c("c_0.54"="0.54","c_0.64"="0.64","c_0.7"="0.7",
                            "c_0.8"="0.8","c_0.94"="0.94")) +
  labs(x="Threshold",y="Precision") +
  #theme(text=element_text(size=20))
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title=element_blank(), 
        legend.position = "bottom", 
        legend.key.size =  unit(0.4, "in"),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA))


# ggplot(precision_df_long, aes(x=Pred_Score_cutoff, y=Precision)) +
#   geom_violin()

## Make a summary of number of unclassified reads by different cutoffs
count_df<-data.frame(matrix(ncol=3,nrow=0))
colnames(count_df)<-c('Threshold','Type','Count')
for(cutoff in c(0.54,0.64,0.7,0.8,0.94)){
  fraction<-dim(a[a$V4<=cutoff,])[1]/dim(a)[1]
  #-record classified fraction
  count_df[nrow(count_df)+1,1:2]<-c(cutoff,'Classified')
  count_df[nrow(count_df),3]<-1-fraction
  #-record unclassified fraction
  count_df[nrow(count_df)+1,1:2]<-c(cutoff,'Unclassified')
  count_df[nrow(count_df),3]<-fraction
}

#--Plot Figure 5B
plot_2 <- ggplot(data=count_df, aes(x=factor(Threshold), y=Count, 
                          fill=factor(Type,levels=c("Unclassified","Classified")))) +
  geom_bar(stat="identity", color="black", width=0.5) +
  scale_fill_manual(values = c("white","grey")) +
  labs(fill="",x="Threshold",y="Fraction of Reads") +
#  theme(text=element_text(size=20), legend.position="top")
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title=element_blank(),legend.position="top",
        legend.key.size =  unit(0.4, "in"),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA))

tiff(file.path(paste(input_dir, "Figure5_score_distribution.tiff", sep="/")), units="in", width=8, height=6, res=300)
ggarrange(plot_1, plot_2, labels = c("A", "B"), ncol=2, nrow=1)
dev.off()

##### 5. ###########
# library("readxl")
# df<-read_excel('confusion-matrix-0.8.xlsx',sheet='species')
# colnames(df)[1]<-'species_pred'
# df_scaled<-df
# for(c in colnames(df[,2:640])){
#   df_scaled[,c]<-scale(df[,c])
# }
# df_long<-pivot_longer(df_scaled,cols=2:640,names_to="species_gt")
# ggplot(df_long,aes(species_gt,species_pred,fill=value)) +
#   geom_tile() +
#   scale_fill_gradient(low="white", high="blue") +
#   theme(axis.text = element_text(size = 1),
#         axis.text.x = element_text(angle=90,hjust=1))

##### 6. ###########
# check the distribution of precision and get a list of outliers
# q<-quantile(as.numeric(pmtx_0.94[,2]), probs=c(0.25,0.75))
# cout<-q[1]-1.5*(q[2]-q[1])
# outliers<-pmtx_0.94[pmtx_0.94[,2]<cout,1]


##### 7. ###########
# df_files<-data.frame(
#   fs=c(
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-species-confidence-scores.tsv",
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-genus-confidence-scores.tsv",
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-family-confidence-scores.tsv",
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-order-confidence-scores.tsv",
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-class-confidence-scores.tsv",
#     "updated-testing-set-all-out-ncbi-cutoff-0.0-phylum-confidence-scores.tsv"
#   ),
#   taxlevel=c(
#     "Species",
#     "Genus",
#     "Family",
#     "Order",
#     "Class",
#     "Phylum"
#   )
# )
# 
# df_scores<-data.frame(matrix(ncol=4,nrow=0))
# for(i in 1:dim(df_files)[1]){
#   f<-df_files[i,]$fs
#   t<-df_files[i,]$taxlevel
#   a <- read.table(file=f, sep="\t")
#   a$V3<-rep(t,dim(a)[1])
#   df_scores<-rbind(df_scores,a)
#   
#   # pdf(paste(t,"_correct.pdf",sep=""),width=4,height=2)
#   # ggplot(data=a[a$V2=='correct',], 
#   #        mapping=aes(x=V4,after_stat(count))) +
#   #   geom_histogram(alpha = 0.6, binwidth = 0.01, fill="black") +
#   #   #  scale_y_log10() +
#   #   #labs(x = "Prediction Score", y = "Count",
#   #   labs(x = "", y = "",
#   #        title = "") +
#   #   theme_bw() +
#   #   facet_wrap( ~ V3, ncol=1)
#   # dev.off()
#   # 
#   # pdf(paste(t,"_incorrect.pdf",sep=""),width=4,height=2)
#   # ggplot(data=a[a$V2=='incorrect',], 
#   #        mapping=aes(x=V4,after_stat(count))) +
#   #   geom_histogram(alpha = 0.6, binwidth = 0.01, fill="black") +
#   #   #  scale_y_log10() +
#   #   #labs(x = "Prediction Score", y = "Count",
#   #   labs(x = "", y = "",
#   #        title = "") +
#   #   theme_bw() +
#   #   facet_wrap( ~ V3, ncol=1)
#   # dev.off()
# }
# 
# #--stats of probability distributions
# pstat<-data.frame(matrix(nrow=0,ncol=4))
# colnames(pstat)<-c("Taxonomy_rank","Type","Stat_type","Value")
# for(tr in df_files$taxlevel){
#   for(type in c("correct","incorrect")){
#     x<-quantile(df_scores[df_scores$V2==type & df_scores$V3==tr,]$V4)
#     #-minimum
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"minimum")
#     pstat[nrow(pstat),4]<-x[1]
#     #-maximum
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"maximum")
#     pstat[nrow(pstat),4]<-x[5]
#     #-median
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"median")
#     pstat[nrow(pstat),4]<-x[3]
#     #--25th percentile
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"quatile1")
#     pstat[nrow(pstat),4]<-x[2]
#     #--75th percentile
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"quatile3")
#     pstat[nrow(pstat),4]<-x[4]
#     #--mean
#     pstat[nrow(pstat)+1,1:3]<-c(tr,type,"mean")
#     pstat[nrow(pstat),4]<-
#       mean(df_scores[df_scores$V2==type & df_scores$V3==tr,]$V4)
#   }
# }
# 
# write.table(pstat,file="probability_stats.tsv",quote=F,sep="\t",row.names = F)
# 
# #--number of correct vs incorrect for each taxonomic rank
# x<-aggregate(df_scores$V4,by=list(df_scores$V2,df_scores$V3),FUN=length)
# colnames(x)<-c("Type","Taxonomy_rank","Count")
# for(tr in df_files$taxlevel){
#   cnt_correct<-x[x$Type=="correct"&x$Taxonomy_rank==tr,]$Count
#   cnt_incorrect<-x[x$Type=="incorrect"&x$Taxonomy_rank==tr,]$Count
#   x[nrow(x)+1,1:2]<-c("Correct_fraction", tr)
#   x[nrow(x),3]<-cnt_correct/(cnt_correct+cnt_incorrect)
# }
# 
# write.table(x,file="correct_vs_incorrect_cnt.tsv",quote=F,sep="\t",row.names = F)
# 
# #--Fig3
# ggplot(data=df_scores[df_scores$V3=="Species",], mapping=aes(x=V2,y=V4)) +
#   geom_violin() +
#   labs(x="",y="Probability Score") +
#   theme(text=element_text(size=20)) +
#   facet_wrap( ~ V3, ncol=6)
# # ggplot(data=df_scores[df_scores$V2=='correct',], 
# #        mapping=aes(x=V4,after_stat(count))) +
# #   geom_histogram(alpha = 0.5, binwidth = 0.01, fill="grey") +
# # #  scale_y_log10() +
# #   labs(x = "Prediction Score", y = "Count",
# #        title = "") +
# #   theme_bw() +
# #   facet_wrap( ~ V3, ncol=1)
# # 
# # ggplot(data=df_scores[df_scores$V2=='incorrect',], 
# #        mapping=aes(x=V4,after_stat(count))) +
# #   geom_histogram(alpha = 0.5, binwidth = 0.01, fill="grey") +
# #   #  scale_y_log10() +
# #   labs(x = "Prediction Score", y = "Count",
# #        title = "") +
# #   theme_bw() +
# #   facet_wrap( ~ V3, ncol=1)
# 
# 
# ##### 7. ###########
# accuracy<-read.table(file="accuracy-summary-subsets-testing-wo-unc.tsv", sep="\t")
# tls<-unique(accuracy$V1)
# tranks<-unique(accuracy$V2)
# df<-data.frame(matrix(ncol=length(tls),nrow=length(tranks)))
# rownames(df)<-tranks
# colnames(df)<-tls
# for(tl in tls){
#   for(tr in tranks){
#     df[tr,tl]<-median(accuracy[accuracy$V1==tl & accuracy$V2==tr,]$V3)
#   }
# }
# 
# write.table(df,file="median_accuracy.tsv",quote=F,sep="\t")
# 
# ggplot(data=accuracy, mapping=aes(x=V1,y=V3)) +
#   geom_violin() +
#   labs(x="",y="Accuracy") +
#   theme(text=element_text(size=20),
#         axis.text.x = element_text(angle=90,hjust=1)) +
#   facet_wrap( ~ V2, ncol=6)
