library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
input_dir = args[1]

# load and parse data
values <- c()
type <- c()
group <- c()
for(r in c("species", "genus", "family", "order", "class", "phylum")){
  data <- read.csv(file.path(paste(input_dir, '/all-samples-', r, "-metrics.tsv", sep="")), sep='\t', header=TRUE)
  data <- head(data, -4)
  values <- c(values, as.double(data$recall), as.double(data$precision), as.double(data$F1))
  type <- c(type, rep("Recall",length(data$recall)), rep("Precision",length(data$precision)), rep("F1",length(data$F1)))
  group <- c(group, rep(r,length(data$recall)+length(data$precision)+length(data$F1)))
}

# create dataframe
df <- data.frame(group,type,values)
df$group <- factor(df$group, levels = unique(df$group))

# create facet plots
tiff(file.path(input_dir, "metrics-plot.tiff", sep=""), units="px", width=3500, height=3500, res=300)
bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type)) + ylab("") + xlab("") + 
  scale_fill_manual(values=c("#202020", "#606060", "#A0A0A0"), name="") + theme(panel.spacing = unit(1, "lines"), axis.text=element_text(size = 25), text=element_text(size = 28), panel.border=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + 
  coord_cartesian(ylim=c(0,1))
bp + facet_grid(.~group)
dev.off()


