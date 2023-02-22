options(scipen = 999)
library(ggplot2)
library(stringr)
library(ggpubr)
library(scales)
library(cowplot)
library(grid)
library(dplyr)


args = commandArgs(trailingOnly = TRUE)
oral_microbiome_dir = args[1]
soil_microbiome_dir = args[2]
oral_microbiome_nr = as.numeric(args[3])
soil_microbiome_nr = as.numeric(args[4])
cutoff = args[5]
rank = args[6]
rel_ab = as.numeric(args[7])


#num_reads = 3417111096 # human oral microbiome
#num_reads = 52290557 # soil metagenome

color_palette <- function(rank, cutoff){
  # get list of taxa at given rank
  taxa <- c()
  for(p in c(oral_microbiome_dir, soil_microbiome_dir)){
    dl_toda_data <- read.csv(file.path(paste(p, '/dl-toda/cutoff-', cutoff, '-taxa_profile/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
    dl_toda_taxa <- c(dl_toda_data$V1)
    kraken2_data <- read.csv(file.path(paste(p, '/kraken2/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
    kraken2_taxa <- c(kraken2_data$V1)
    centrifuge_data <- read.csv(file.path(paste(p, '/centrifuge/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
    centrifuge_taxa <- c(centrifuge_data$V1)
    taxa <- c(taxa,dl_toda_taxa)
    taxa <- c(taxa,kraken2_taxa)
    taxa <- c(taxa,centrifuge_taxa)
  }
  taxa <- unique(taxa)
  cat(rank, "\t", length(unique(taxa)))
  # randomly choose colors (for the future generate a set of colors for the number of taxa)
  colors <- c()
  for(t in taxa){
    rgb_vector <- c()
    for(i in 1:3){
      rgb_vector <- c(rgb_vector,sample(seq(from=0,to=1,by=0.0002),1,replace=FALSE))
    }
    colors <- c(colors, rgb(rgb_vector[1],rgb_vector[2],rgb_vector[3]))
  }
  # save colors to csv file
  write.csv(data.frame(taxa,colors),file=file.path(paste('colors_', rank, ".csv", sep="")),row.names=F,col.names=F)
}


# create color palette
#for(r in c("species","genus","class")){
#  color_palette(r, cutoff)
#}

# load data
dl_toda_oral_data <- read.csv(file.path(paste(oral_microbiome_dir, '/dl-toda/cutoff-', cutoff, '-taxa_profile/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
kraken2_oral_data <- read.csv(file.path(paste(oral_microbiome_dir, '/kraken2/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
centrifuge_oral_data <- read.csv(file.path(paste(oral_microbiome_dir, '/centrifuge/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
dl_toda_soil_data <- read.csv(file.path(paste(soil_microbiome_dir, '/dl-toda/cutoff-', cutoff, '-taxa_profile/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
kraken2_soil_data <- read.csv(file.path(paste(soil_microbiome_dir, '/kraken2/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)
centrifuge_soil_data <- read.csv(file.path(paste(soil_microbiome_dir, '/centrifuge/taxa_profile_', rank, sep="")), sep='\t', header=FALSE)

# update centrifuge data to take into account pairs of reads
centrifuge_oral_data$V2 <- centrifuge_oral_data$V2*2
centrifuge_soil_data$V2 <- centrifuge_soil_data$V2*2

# combine taxa with percentage below 1%
update_data <- function(df, tool, data, num_reads){
  # get relative abundance of unknown taxa
  na_ra <- df$V2[df$V1 == 'na']/num_reads*100
  # remove na/unknown
  df <- df[df$V1 !='na',]
  sum_ra <- sum(df$V2/num_reads*100)
  n_taxa <- dim(df)[1]
  # compute percentages per taxon
  df$V2 <- df$V2/num_reads*100
  sum_ra_above <- sum(df$V2[df$V2 >= rel_ab])
  # get percentages above or equal to threshold
  df_above_th <- df[df$V2 >= rel_ab,]
  sum_ra_below <- 0
  if(rel_ab >= 0){
    # sum percentages below threshold
    sum_below_th <- sum(df$V2[df$V2 < rel_ab])
    #df_above_th[nrow(df_above_th)+1,] = c("other", sum_below_th)
    sum_ra_below <- sum_below_th
  }
  df_above_th$V2 <- as.numeric(df_above_th$V2)
  #"\t", dim(df_above_th[df_above_th$V1 != "other",])[1],
  cat(tool, "\t", data, "\t", n_taxa, "\t", sum_ra, "\t", dim(df_above_th)[1], "\t",  sum_ra_above, "\t", sum_ra_below, "\t", na_ra, "\n")
  return(df_above_th)
  #return(df)
}

# compute relative abundances
dl_toda_oral_df <- update_data(dl_toda_oral_data, "DL-TODA\t", "oral\t", oral_microbiome_nr)
kraken2_oral_df <- update_data(kraken2_oral_data, "Kraken2\t", "oral\t", oral_microbiome_nr)
centrifuge_oral_df <- update_data(centrifuge_oral_data, "Centrifuge\t", "oral\t", oral_microbiome_nr)
dl_toda_soil_df <- update_data(dl_toda_soil_data, "DL-TODA\t", "soil\t", soil_microbiome_nr)
kraken2_soil_df <- update_data(kraken2_soil_data, "Kraken2\t", "soil\t", soil_microbiome_nr)
centrifuge_soil_df <- update_data(centrifuge_soil_data, "Centrifuge\t", "soil\t", soil_microbiome_nr)

# write majority taxa predicted by DL-TODA, Kraken2 and Centrifuge
write.csv(dl_toda_oral_df[order(dl_toda_oral_df$V2,decreasing=FALSE),],file=file.path(paste("dl_toda_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_oral.csv", sep = "")), row.names=FALSE, col.names=FALSE)
write.csv(kraken2_oral_df[order(kraken2_oral_df$V2,decreasing=FALSE),],file=file.path(paste("kraken2_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_oral.csv", sep = "")), row.names=FALSE, col.names=FALSE)
write.csv(centrifuge_oral_df[order(centrifuge_oral_df$V2,decreasing=FALSE),],file=file.path(paste("centrifuge_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_oral.csv", sep = "")), row.names=FALSE, col.names=FALSE)
write.csv(dl_toda_soil_df[order(dl_toda_soil_df$V2,decreasing=FALSE),],file=file.path(paste("dl_toda_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_soil.csv", sep = "")), row.names=FALSE, col.names=FALSE)
write.csv(kraken2_soil_df[order(kraken2_soil_df$V2,decreasing=FALSE),],file=file.path(paste("kraken2_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_soil.csv", sep = "")), row.names=FALSE, col.names=FALSE)
write.csv(centrifuge_soil_df[order(centrifuge_soil_df$V2,decreasing=FALSE),],file=file.path(paste("centrifuge_taxa_cutoff_", cutoff, "_", rel_ab, "_", rank, "_soil.csv", sep = "")), row.names=FALSE, col.names=FALSE)

# load colors
colors <- read.csv(file.path(paste('colors_', rank, ".csv", sep="")),sep=",",header=TRUE)
cat("colors", "\t", dim(colors), "\n")

# create dataframe
create_df <- function(dl_toda_df, centrifuge_df, kraken2_df){
  df <- rbind(dl_toda_df, centrifuge_df, kraken2_df)
  df$V3 <- c(rep("DL-TODA", dim(dl_toda_df)[1]), rep("Centrifuge", dim(centrifuge_df)[1]), rep("Kraken2", dim(kraken2_df)[1]))
  # order data based on tool
  df$V3 <- factor(df$V3, levels = unique(df$V3))
  ## order data based on taxa
  ##df$V1 <- factor(df$V1, levels=taxa)
  colnames(df) <- c('taxa','value','tool')
  df$value <- as.numeric(df$value)
  
  return(df)
}

oral_df <- create_df(dl_toda_oral_df, centrifuge_oral_df, kraken2_oral_df)
soil_df <- create_df(dl_toda_soil_df, centrifuge_soil_df, kraken2_soil_df)

# get vector of colors
get_colors <- function(df){
  # add colors of taxa sorted alphabetically
  taxa_sorted <- str_sort(unique(df$taxa))
  df_colors <- c()
  for(t in taxa_sorted){
    df_colors <- c(df_colors, colors$colors[colors$taxa==t])
  }
  return(df_colors)
}

oral_df_colors <- get_colors(oral_df)
soil_df_colors <- get_colors(soil_df)

# create stacked plot
create_stacked_plot <- function(df, data, df_colors){
  plot <- ggplot(df, aes(y=value, x=tool, fill=taxa)) + geom_bar(position='stack', stat='identity', colour="black") +
    scale_fill_manual(values = df_colors)+
    theme(legend.text = element_text(size=5),
          legend.key.height= unit(0.5, 'cm'),
          legend.key.width= unit(0.5, 'cm')) +
    ylab("Percentage of reads") + 
    xlab("")

  return(plot)
}

plot1 <- create_stacked_plot(oral_df, 'oral', oral_df_colors)
plot2 <- create_stacked_plot(soil_df, 'soil', soil_df_colors)


tiff(file.path(paste("legend_cutoff_", cutoff, "_oral_", rel_ab, "_", rank, ".tiff", sep = "")), units="in", width=10, height=6, res=300)
legend1 <- cowplot::get_legend(plot1)
ggarrange(legend1, ncol=1, nrow=1)
dev.off()

tiff(file.path(paste("legend_cutoff_", cutoff, "_soil_", rel_ab, "_", rank, ".tiff", sep = "")), units="in", width=10, height=6, res=300)
legend2 <- cowplot::get_legend(plot2)
ggarrange(legend2, ncol=1, nrow=1)
dev.off()

plot1 <- plot1 + theme(axis.title = element_text(size = 12),
                     axis.text = element_text(size = 12),legend.position = 'none')

plot2 <- plot2 + theme(axis.title = element_text(size = 12),
                       axis.text = element_text(size = 12),legend.position = 'none')

tiff(file.path(paste("stacked_plot_cutoff_", cutoff, "_", rel_ab, "_", rank, ".tiff", sep = "")), units="in", width=8.5, height=6, res=300)
ggarrange(plot1, plot2, labels = c("A", "B"), ncol=2, nrow=1)
dev.off()





#get_outliers <- function(df, maxval){
#  # get outliers
#  dd <- df %>% filter(value>maxval) %>%
#    group_by(tool) %>%
#    summarise(outlier_txt=paste(sort(round(value,digits=2)),collapse="\n"))
#  return(dd)
#}

# create histogram of relative abundances
#maxval=0
#if(rank == 'species'){
#  maxval = 0.3
#}
#if(rank == 'genus'){
#  maxval = 0.4
#}
#if(rank == 'class'){
#  maxval = 0.7
#}

#outliers <- get_outliers(df, maxval)
#print(outliers)
#tiff(file.path(paste(input_dir, '/', "bp_cutoff_", cutoff, "_", rel_ab, "_", rank, ".tiff", sep = "")), units="in", width=5.5, height=6, res=300)
#ggplot(df, aes(y=value, fill=tool, x=tool)) + geom_boxplot() + ylab("") + xlab("") + 
#  scale_fill_manual(values=c("#000000", "#434343", "#999999"), name="") + 
#  theme(panel.spacing = unit(1, "lines"), 
#        plot.margin = margin(t=10,r=10,b=10,l=30), 
#        axis.title.y=element_text(size = 18, vjust=5), 
#        axis.text=element_text(size = 18), 
#        text=element_text(size = 18), 
#        panel.border=element_blank(), 
#        axis.ticks.x=element_blank(), 
#        axis.text.x=element_blank(), 
#        axis.title.x=element_blank(), 
#        legend.title = element_blank(),
#        legend.position = "bottom")+
#  coord_cartesian(ylim=c(0,maxval)) +
#  geom_text(data=outliers,aes(x=tool, y=maxval,label=outlier_txt), size=3.0,vjust=0.90,hjust=1.5) + 
#  geom_segment(data=outliers, aes(x=tool, xend=tool, y=maxval*0.95, yend=maxval), arrow = arrow(length = unit(0.3,"cm")))
#dev.off()

#outliers <- get_outliers(df, maxval)
#print(outliers)
#tiff(file.path(paste(input_dir, '/', "hist_cutoff_", cutoff, "_", rel_ab, "_", rank, ".tiff", sep = "")), units="in", width=5.5, height=6, res=300)
#ggplot(df, aes(x=value)) + geom_histogram() + ylab("Number of taxa") + xlab("Relative abundance") +
#  theme(axis.title = element_text(size = 12),
#        axis.text = element_text(size = 12)) +
#  coord_cartesian(xlim=c(0,5)) +
#  scale_x_continuous(breaks=seq(0, max(df$value), by = 1))
#dev.off()

