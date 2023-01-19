library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
input_file = args[1]
output_dir = args[2]

data <- read.csv(input_file, sep='\t', header=FALSE)

tiff(file.path(paste(output_dir, 'FigureS2_precision_training_reads.tiff', sep = "/")), units="in", width=8.5, height=6, res=300)
ggplot(data, aes(x=log(V2), y=V3)) + geom_point(position="jitter") +
  ylab("Precision") + 
  xlab("Number of training reads (natural log)") +
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey'),
        panel.border = element_rect(colour="light grey", fill=NA))