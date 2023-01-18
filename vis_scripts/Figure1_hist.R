library(ggplot2)
library(stringr)
library(ggpubr)
library(scales)

args = commandArgs(trailingOnly = TRUE)
train_input_file = args[1]
test_input_file = args[2]
output_dir = args[3]

train_data <- read.csv(train_input_file, sep='\t', header=FALSE)
test_data <- read.csv(test_input_file, sep='\t', header=FALSE)


cat("training - ", "median: ", median(train_data$V2), " min: ", min(train_data$V2), " max: ", max(train_data$V2), "\n")
cat("testing - ", "median: ", median(train_data$V2), " min: ", min(train_data$V2), " max: ", max(train_data$V2), "\n")


# plot histogram of read count per species for training and testing datasets
plot1 <- ggplot(train_data, aes(x=log(V2))) + geom_histogram() +
  ylab("Number of species") + 
  xlab("Number of training reads per species (natural log)") +
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA)) +
  scale_x_continuous(breaks = pretty_breaks())


plot2 <- ggplot(test_data, aes(x=log(V2))) + geom_histogram() +
  ylab("Number of species") + 
  xlab("Number of testing reads per species (natural log)") +
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA)) +
  scale_x_continuous(breaks = pretty_breaks())


plot_list <- c(plot1, plot2)
tiff(file.path(paste(output_dir, 'train_test_hist.tiff', sep = "/")), units="in", width=8.5, height=6, res=300)
ggarrange(plot1, plot2, labels = c("A", "B"), ncol=2, nrow=1)
dev.off()
