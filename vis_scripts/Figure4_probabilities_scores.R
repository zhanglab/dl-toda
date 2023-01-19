library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
input_file = args[1]
output_dir = args[2]

# load data
input_df <- read.csv(input_file, sep = "\t", header=FALSE)
input_df$species = ifelse(input_df$V1 == input_df$V2, 'Correct',
                      ifelse(input_df$V1 != input_df$V2, 'Incorrect', 'None'))

input_df$genus = ifelse(input_df$V3 == input_df$V4, 'Correct',
                            ifelse(input_df$V3 != input_df$V4, 'Incorrect', 'None'))

input_df$family = ifelse(input_df$V5 == input_df$V6, 'Correct',
                  ifelse(input_df$V5 != input_df$V6, 'Incorrect', 'None'))

input_df$order = ifelse(input_df$V7 == input_df$V8, 'Correct',
                  ifelse(input_df$V7 != input_df$V8, 'Incorrect', 'None'))

input_df$class = ifelse(input_df$V9 == input_df$V10, 'Correct',
                  ifelse(input_df$V9 != input_df$V10, 'Incorrect', 'None'))

input_df$phylum = ifelse(input_df$V11 == input_df$V12, 'Correct',
                  ifelse(input_df$V11 != input_df$V12, 'Incorrect', 'None'))



# create dataframe
values<- c(rep(input_df$V13, 6))
group <- c(rep("phylum",length(input_df$V13)), rep("class",length(input_df$V13)), rep("order",length(input_df$V13)), rep("family",length(input_df$V13)), rep("genus",length(input_df$V13)), rep("species",length(input_df$V13)))
type <- c(input_df$phylum, input_df$class, input_df$order, input_df$family, input_df$genus, input_df$species)
df <- data.frame(group,type,values)
df$group <- factor(df$group, levels = unique(df$group))



# create boxplot
tiff(file.path(paste(output_dir, '/', "DL_TODA_prob_scores.tiff", sep = "")), units="in", width=4.5, height=6, res=300)
#tiff(file.path(paste(output_dir, '/', "DL_TODA_prob_scores.tiff", sep = "")), type="cairo", units="in", width=4.5, height=6, res=300)
bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type), outlier.shape=NA) + ylab("Probability Score") + xlab("") + 
  scale_fill_manual(values=c("#dedede","#A0A0A0"), name="") + 
  theme(panel.spacing = unit(1, "lines"), 
        axis.text=element_text(size = 12), 
        axis.title=element_text(size = 12), 
        panel.border=element_rect(colour="light grey", fill=NA), 
        axis.ticks.x=element_blank(), 
        axis.text.x=element_blank(), 
        axis.title.x=element_blank(), 
        legend.position = "bottom", 
        legend.key.size =  unit(0.3, "in"),
        legend.text=element_text(size=12),
        panel.background = element_rect(fill='white'),
        panel.grid.major = element_line(color='light grey'),
        panel.grid.minor = element_line(color='light grey')) + 
  coord_cartesian(ylim=c(0,1))
# create facet plot
bp + facet_grid(.~group)
dev.off()


# create violin plots
#tiff(file.path(paste(input_dir, '/', "violin-plot-cutoff-0.0-prob.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#bp <- ggplot(df, aes(y=values, x=type)) + geom_violin(aes(fill=type)) + ylab("Prediction score") + xlab("") + 
#  scale_fill_manual(values=c("#202020","#A0A0A0"), name="") + 
#  theme(panel.spacing = unit(1, "lines"),
#        axis.text=element_text(size = 25), 
#        text=element_text(size = 28), 
#        panel.border=element_blank(), 
#        axis.ticks.x=element_blank(), 
#        axis.text.x=element_blank(), 
#        axis.title.x=element_blank(), 
#        legend.position = "bottom", 
#        legend.key.size =  unit(0.5, "in")) + 
#  coord_cartesian(ylim=c(0,1))
#bp + facet_grid(.~group)
#dev.off()

# create facet plots
#tiff(file.path(paste(input_dir, '/', "boxplot-cutoff-0.0-prob-w-outliers-triangle-17-grey-0.001.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type),outlier.shape=17, outlier.alpha=0.001, outlier.colour="black", outlier.fill="grey") + ylab("Probability Score") + xlab("") + 
#  scale_fill_manual(values=c("#dedede","#A0A0A0"), name="") + theme(panel.spacing = unit(1, "lines"), axis.text=element_text(size = 25), text=element_text(size = 28), panel.border=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + 
#  coord_cartesian(ylim=c(0,1))
# create facet plot
#bp + facet_grid(.~group)
#dev.off()

# create facet plots
#tiff(file.path(paste(input_dir, '/', "boxplot-cutoff-0.0-prob-w-outliers-triangle-17-grey-0.01.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type),outlier.shape=17, outlier.alpha=0.01, outlier.colour="black", outlier.fill="grey") + ylab("Probability Score") + xlab("") + 
#  scale_fill_manual(values=c("#dedede","#A0A0A0"), name="") + theme(panel.spacing = unit(1, "lines"), axis.text=element_text(size = 25), text=element_text(size = 28), panel.border=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + 
#  coord_cartesian(ylim=c(0,1))
# create facet plot
#bp + facet_grid(.~group)
#dev.off()

# create facet plots
#tiff(file.path(paste(input_dir, '/', "boxplot-cutoff-0.0-prob-w-outliers-triangle-17-0.01.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type),outlier.shape=17, outlier.alpha=0.01, outlier.colour="black", outlier.fill="red") + ylab("Probability Score") + xlab("") + 
#  scale_fill_manual(values=c("#dedede","#A0A0A0"), name="") + theme(panel.spacing = unit(1, "lines"), axis.text=element_text(size = 25), text=element_text(size = 28), panel.border=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + 
#  coord_cartesian(ylim=c(0,1))
# create facet plot
#bp + facet_grid(.~group)
#dev.off()

# create facet plots
#tiff(file.path(paste(input_dir, '/', "boxplot-cutoff-0.0-prob-w-outliers-triangle-17-0.001.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#bp <- ggplot(df, aes(y=values, fill=group)) + geom_boxplot(aes(fill=type),outlier.shape=17, outlier.alpha=0.001, outlier.colour="black", outlier.fill="red") + ylab("Probability Score") + xlab("") + 
#  scale_fill_manual(values=c("#dedede","#A0A0A0"), name="") + theme(panel.spacing = unit(1, "lines"), axis.text=element_text(size = 25), text=element_text(size = 28), panel.border=element_blank(), axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x=element_blank(), legend.position = "bottom", legend.key.size =  unit(0.5, "in")) + 
#  coord_cartesian(ylim=c(0,1))
# create facet plot
#bp + facet_grid(.~group)
#dev.off()


# create bar chart with percentage of incorrect and correct predictions
# create dataframe
#df_pred <- data.frame(values=c(df_species$V1, df_genus$V1, df_family$V1, df_order$V1, df_class$V1), group=c(rep("species",length(df_species$V1)), rep("genus",length(df_genus$V1), rep("family",length(df_family$V1)), rep("order",length(df_order$V1)), rep("class",length(df_class$V1)))))
#tiff("/Users/cecilecres/Desktop/fraction-pred.tiff", units="px", width=3500, height=3500, res=300)
#ggplot(df_pred, aes(x=values, fill=group)) + geom_bar(aes(y = (..count..)/sum(..count..))) + scale_y_continuous(labels=scales::percent) + ylab("relative frequencies") + xlab("") + scale_fill_manual(values=c("#DC5B57","#008CEE"), name="") + theme(axis.text = element_text(size = 10), text = element_text(size = 12))   
#dev.off()

# create histogram of probabilities for correct and incorrect predictions
#tiff(file.path(paste(input_dir, '/', rank, "-cutoff-0.0-hist-prob.tiff", sep = "")), type="cairo", units="px", width=3500, height=3500, res=300)
#ggplot(df, aes(x=df$V4, fill=df$V2)) + ylab("count") + xlab("confidence scores") + coord_cartesian(xlim=c(0,1)) + scale_fill_manual(values=c("#DC5B57","#008CEE"), name="") + geom_histogram(position = "identity", alpha = 0.4, bins = 30) + theme(axis.text = element_text(size = 10), text = element_text(size = 12))
#dev.off()
 
