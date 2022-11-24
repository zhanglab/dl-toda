library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
input_dir = args[1]

# load data
training_data <- read.csv(list.files(path=input_dir, pattern="training_data", full.names=FALSE), sep='\t', header=FALSE)
validation_data <- read.csv(list.files(path=input_dir, pattern="validation_data", full.names=FALSE), sep='\t', header=FALSE)
training_info <- read.csv(list.files(path=input_dir, pattern="training-summary", full.names=FALSE), sep='\t', header=FALSE)

# determine x axis labels
num_batch_per_epoch = training_info$V2[13]
num_epochs = training_info$V2[3]
x_axis_breaks = seq(from = num_batch_per_epoch, to = num_batch_per_epoch*num_epochs, by = num_batch_per_epoch)
x_axis_ticks = seq(from = 1, to = num_epochs, by = 1)
linetypes <- c("Training" = "solid", "Validation" = "dashed")

# plot training and validation accuracy
tiff("accuracy.tiff", units="px", width=3500, height=3500, res=300)
ggplot() + geom_line(data=training_data, aes(x=V2, y=V5, linetype="Training"), size=1) +
  geom_line(data=validation_data, aes(x=V2, y=V4, linetype="Validation"), size=1) + geom_vline(xintercept=14,lwd=2,colour="black") +
  xlab("Epoch") + ylab("Accuracy") + theme(axis.text=element_text(size = 25), text=element_text(size = 28), legend.title=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + coord_cartesian(ylim=c(min(validation_data$V4,training_data$V5),1)) +
  scale_x_continuous(breaks=x_axis_breaks, labels=x_axis_ticks) + scale_linetype_manual(values = linetypes)
dev.off()

# plot training and validation loss
tiff("loss.tiff", units="px", width=3500, height=3500, res=300)
ggplot() + geom_line(data=training_data, aes(x=V2, y=V4, linetype="Training"), size=1) +
  geom_line(data=validation_data, aes(x=V2, y=V3, linetype="Validation"), size=1) + geom_vline(xintercept=14,lwd=2,colour="black") +
  xlab("Epoch") + ylab("Loss") + theme(axis.text=element_text(size = 25), text=element_text(size = 28), legend.title=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + coord_cartesian(ylim=c(min(validation_data$V3,training_data$V4),max(validation_data$V3,training_data$V4)+0.5)) +
  scale_x_continuous(breaks=x_axis_breaks, labels=x_axis_ticks) + scale_linetype_manual(values = linetypes)
dev.off()

