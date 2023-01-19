library(ggplot2)
library(plyr)

args = commandArgs(trailingOnly = TRUE)
input_filename = args[1]
output_dir = args[2]

# Function to calculate the mean and the standard deviation
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE), median = median(x[[col]], na.rm=TRUE),
      quantile = quantile(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

# load input data
df = read.csv(input_filename, sep= "\t", header=FALSE)

# create dataframe and define columns names
colnames(df) = c("type", "ranks", "values")
# get summary of stats
df2 <- data_summary(df, varname="values", groupnames=c("type", "ranks"))
df2$type <- factor(df2$type, levels = c("DL-TODA","Centrifuge", "Kraken2"))
df2$ranks <- factor(df2$ranks, levels = c("phylum","class","order","family","genus","species"))

# create barplot figure
figname = file.path(paste(output_dir, 'Figure6_accuracy.tiff', sep = "/"))
tiff(figname, units="in", width=4.5, height=6, res=300)
ggplot(data=df2, aes(x=ranks, y=values, fill=type, order=type)) + geom_bar(stat='identity', position='dodge', width=0.6) +
  geom_errorbar(aes(ymin=values-sd, ymax=values+sd), width=.5, position=position_dodge()) + ylab("Accuracy") + xlab("") + 
  scale_fill_manual(values=c("#202020", "#606060", "#A0A0A0"), name="") +
  coord_cartesian(ylim=c(0,1)) + 
  theme(
        axis.text=element_text(size = 12), 
        text=element_text(size = 12), 
        legend.text = element_text(size = 12),
        legend.title=element_blank(), 
        legend.position = "bottom", 
        legend.key.size =  unit(0.3, "in"),
        axis.title.y=element_text(size=12),  
        axis.text.x=element_text(size=12, angle=45, color='black', hjust=1),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA))
dev.off()


