library(ggplot2)
library(ggpubr)

args = commandArgs(trailingOnly = TRUE)
input_dir = args[1]


# load data
training_data <- read.csv(list.files(path=input_dir, pattern="training_data", full.names=FALSE), sep='\t', header=FALSE)
validation_data <- read.csv(list.files(path=input_dir, pattern="validation_data", full.names=FALSE), sep='\t', header=FALSE)
training_info <- read.csv(list.files(path=input_dir, pattern="training-summary", full.names=FALSE), sep='\t', header=FALSE)

# determine x axis labels
num_batch_per_epoch = as.numeric(training_info$V2[16])
num_epochs = as.numeric(training_info$V2[6])

x_axis_breaks = seq(from = num_batch_per_epoch, to = num_batch_per_epoch*num_epochs, by = num_batch_per_epoch*9)
x_axis_ticks = seq(from = 1, to = num_epochs, by = 9)
linetypes <- c("Training" = "solid", "Validation" = "dashed")

print(x_axis_ticks)
print(x_axis_breaks)

colnames(training_data) <- c("epoch", "batch", "learning_rate", "loss", "accuracy")
colnames(validation_data) <- c("epoch", "batch", "loss", "accuracy")

print(training_data)

# plot training and validation accuracy
plot1 <- ggplot() + geom_line(data=training_data, aes(x=batch, y=accuracy, linetype="Training"), size=0.6) +
  geom_line(data=validation_data, aes(x=batch, y=accuracy, linetype="Validation"), size=0.6) + 
  #geom_vline(xintercept=14*num_batch_per_epoch,lwd=0.6,colour="black",linetype="dotted") +
  xlab("Epoch") + ylab("Accuracy") + 
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title=element_blank(), 
        legend.position = "bottom", 
        legend.key.size =  unit(0.4, "in"),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA),
        panel.spacing = unit(1, "cm", data = NULL)) +
  scale_x_continuous(breaks=x_axis_breaks, labels=x_axis_ticks) + scale_linetype_manual(values = linetypes)

# plot training and validation loss
plot2 <- ggplot() + geom_line(data=training_data, aes(x=batch, y=loss, linetype="Training"), size=0.6) +
  geom_line(data=validation_data, aes(x=batch, y=loss, linetype="Validation"), size=0.6) + 
  #geom_vline(xintercept=14*num_batch_per_epoch,lwd=0.6,colour="black",linetype="dotted") +
  xlab("Epoch") + ylab("Loss") + 
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.title=element_blank(), 
        legend.position = "bottom", 
        legend.key.size =  unit(0.4, "in"),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA)) +
  coord_cartesian(ylim=c(min(validation_data$loss,training_data$loss),max(validation_data$loss,training_data$loss)+0.5)) +
  scale_x_continuous(breaks=x_axis_breaks, labels=x_axis_ticks) + scale_linetype_manual(values = linetypes)

tiff("Figure3_learning_curves.tiff", units="in", width=8, height=6, res=300)
ggarrange(plot1, plot2, labels = c("A", "B"), widths=c(1, -0.05, 1), ncol=2, nrow=1)
dev.off()
