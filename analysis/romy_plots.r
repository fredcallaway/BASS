
##### libraries ####
library(languageR)
library(MASS)
library(ggplot2) #cookbook for R/ Graphs
library(hexbin)
library(memisc)
library(reshape)
library(reshape2) #melt and cast -> restructure and aggregate data
library(data.table)
library(coin) #for permutation tests
library(psych)
library(doBy)
library(heplots)
library(plyr) #necessary for ddply
library(matrixStats) 
library(foreign) 
library(Hmisc)
library (stringr)
library(gridExtra)
library(grid)
library(gdata)
library(effects)
library(RColorBrewer)
library(sjPlot)
library(ggExtra)
library(Rmisc)

# # get to the right place:
# fileLoc <- dirname(rstudioapi::getSourceEditorContext()$path)
# setwd(fileLoc) # go to script location first
# setwd("../../FredResults") # we're now in your results folder

# basepath <- getwd()
# ######## load empirical data #######

# %% ====================  ====================

basepath = "Romy"
### load data ###
# 3 studies - I will only leave the last one for example plotting.
input_file = '../model/results/qualitative_sim_may6.csv' #'sep9-stupid_confidence.csv'#'sep9-stupid_prior.csv'#sep7-stupid_prior.csv'#'sep7-basic.csv'#'sep7-stupid_confidence.csv'#'sep7-basic.csv'#'sep8-replicate-jul24-D.csv'#'jul24-D.csv' #'jul24-C.csv' #'jul24-B.csv' #'jul24.csv' #'jun29.csv' #
# input_file = '../model/results/qualitative_sim_apr11.csv' #'sep9-stupid_confidence.csv'#'sep9-stupid_prior.csv'#sep7-stupid_prior.csv'#'sep7-basic.csv'#'sep7-stupid_confidence.csv'#'sep7-basic.csv'#'sep8-replicate-jul24-D.csv'#'jul24-D.csv' #'jul24-C.csv' #'jul24-B.csv' #'jul24.csv' #'jun29.csv' #
a1c = read.csv(input_file)

### color definitions ###
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
darkcols <- brewer.pal(8, "RdGy")
darkcolssub <-  darkcols[c(1:2, 6:8)]


darkcolsg <- brewer.pal(8, "Greens")
darkcolssubg <-  darkcolsg[c(4:8)]

curfigpath = "figs/qualitative_may6/"
dir.create(curfigpath, showWarnings=F)
# ### generate an output directory for each datafile
# x <-  str_locate_all(pattern ='.csv',input_file)
# stop = x[[1]][1]-1
# newfolder <- substr(input_file,1, stop)

# dir.create(paste0(basepath,'/Figures/Model/', newfolder))

# curfigpath <- paste0(basepath,'/Figures/Model/', newfolder, '/')

if (exists("risk_aversion", a1c)){
  a1c$risk_aversion_level <- a1c$risk_aversion*1000
}else {
  a1c$risk_aversion_level <- -10
}

a1c$risk_aversion_level <- as.factor(a1c$risk_aversion_level)

a1c$RT = a1c$pt1 + a1c$pt2
for (i in levels(a1c$risk_aversion_level)){
  ## compute relevant variables
  a1c$sfstItemV <- scale(a1c$val1, scale=FALSE, center=TRUE)
  a1c$ssndItemVal <- scale(a1c$val2, scale=FALSE, center=TRUE)
  a1c$fstosnd <- a1c$val1 -a1c$val2 # relative Value (first over second)
  a1c$avV <- (a1c$val1  + a1c$val2)/2 # overall Value (average Value)
  a1c$savV <- scale(a1c$avV, scale=FALSE, center=TRUE) # mean centered OV
  a1c$sVD <- scale(abs(a1c$val1 -a1c$val2), scale=FALSE, center=TRUE) # mean centered OV
  
  
  a1c$RT <- a1c$pt1 + a1c$pt2 # computing RT from PT
  a1c$cRT <- scale(a1c$RT, scale=FALSE, center=TRUE) # centering that again
  a1c$pdfirst <- a1c$pt1/(a1c$pt1 +a1c$pt2)
  a1c$spdfirst<- a1c$pdfirst -0.5#scale(a1c$pdfirst, scale=FALSE, center=TRUE)
  
  ## make choices the way they are in the other script
  a1c$isfirstIchosen <- rep(0, length(a1c$choice))
  a1c$isfirstIchosen[a1c$choice==1] <- 1
  
  a1c$totalConfidence <- scale((a1c$conf1  + a1c$conf2)/2, scale=FALSE, center=TRUE)
  
  a1c$ConfDif <-scale((a1c$conf1-a1c$conf2), scale=FALSE, center=TRUE)
  
  if (exists("over_confidence", a1c)){
    a1c$cConfBias <- scale(a1c$over_confidence, scale=FALSE, center=TRUE)
    doBias= 1
  }
  if (exists("over_confidence_slope", a1c))  {
    if (length(unique(a1c$over_confidence_slope))>1){
      a1c$cConfBias <- scale(a1c$over_confidence_slope, scale=FALSE, center=TRUE)
      doBias= 1
    }else{
      if (length(unique(a1c$over_confidence_intercept))>1){
        a1c$cConfBias <- scale(a1c$over_confidence_intercept, scale=FALSE, center=TRUE)
        doBias= 1 
      }else{
      a1c$cConfBias <-  mean(a1c$conf1  + a1c$conf2)
      doBias= 0}
      }
  }else {
    a1c$cConfBias <-  mean(a1c$conf1  + a1c$conf2)
    doBias= 0
  }
}
   # place holder until we actually vary that for different agents
  
###### Study 503 non-replication ######
### note to myself: when I let RT interact with values and presentation duration, I get a reliable interaction with second item value
# 
# 
# print (summary(Choicemod0 <- glm(isfirstIchosen ~ ((sfstItemV + ssndItemVal)* spdfirst)+cRT,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ]))) 
# 
# 
# eff_df <- Effect(c("sfstItemV", "spdfirst"), Choicemod0, xlevels=list(sfstItemV =seq(min(a1c$sfstItemV ), max(a1c$sfstItemV), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )
# darkcols <- brewer.pal(8, "RdGy")
# darkcolssub <-  darkcols[c(1:2, 6:8)]
# #plot(eff_df)
# IA <- as.data.frame(eff_df)
# IA$spdfirst <- as.factor(round(IA$spdfirst,1))
# IAf<- IA
# IAf$Item <- rep("first", length(IAf$fit))
# IAf <- plyr:::rename(IAf, c("sfstItemV" = "ItemV"))
# 
# eff_df <- Effect(c("ssndItemVal", "spdfirst"), Choicemod0, xlevels=list(ssndItemVal =seq(min(a1c$ssndItemVal ), max(a1c$ssndItemVal), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )
# darkcols <- brewer.pal(8, "RdGy")
# darkcolssub <-  darkcols[c(1:2, 6:8)]
# #plot(eff_df)
# IA <- as.data.frame(eff_df)
# IA$spdfirst <- as.factor(round(IA$spdfirst,1))
# IAs<- IA
# IAs$Item <- rep("second", length(IAs$fit))
# IAs <- plyr:::rename(IAs, c("ssndItemVal" = "ItemV"))
# 
# IAall <- rbind(IAf, IAs)
# 
# plmodfst <- ggplot(data=IAall, aes(x=ItemV, y=fit, linetype= Item, color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IAall, aes(x=ItemV, max = fit + se, min = fit- se, linetype= Item, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
#   scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IAall, aes(x=ItemV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
#   coord_cartesian(ylim = c(0, 1)) 
# 
# pdf(paste0(curfigpath, "RPDbyItemVFredSim",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(plmodfst)
# dev.off()



print (summary(Choicemod0 <- glm(isfirstIchosen ~ cRT,family=binomial(link='logit') , data = a1c)))

xlevels = list(cRT =seq(min(a1c$cRT, na.rm=TRUE ), max(a1c$cRT, na.rm=TRUE), 0.1))
eff_df <- effect(c("cRT"), Choicemod0, xlevels=xlevels)

IA <- as.data.frame(eff_df)
IA$cRT <- IA$cRT - min(a1c$cRT, na.rm=TRUE )

pRT<- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(curfigpath, "RTonBiasFredSim",i,".pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(pRT)
dev.off()

print (summary(Choicemod0 <- glm(isfirstIchosen ~ (fstosnd + savV)* spdfirst +cRT,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ])))

eff_df <- Effect(c("fstosnd"), Choicemod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd, na.rm=TRUE ), max(a1c$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)

(pRV<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) )

pdf(paste0(curfigpath, "RVFredSim",i,".pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(pRV)
dev.off()

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

(plmodfstOV <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) )

pdf(paste0(curfigpath, "RPDbyOV504",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(plmodfstOV)
dev.off()

###### Certainty analyses ######

darkcolsg <- brewer.pal(8, "Greens")
darkcolssubg <-  darkcolsg[c(4:8)]

################################ 3 -facets of Confidence #######################

## fit properly!
if (doBias==1 ){
print (summary(Choicemod0 <- glm(isfirstIchosen ~ ((fstosnd+savV)*(totalConfidence+ConfDif+cConfBias+spdfirst)) ,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ]))) 
}else{
print (summary(Choicemod0 <- glm(isfirstIchosen ~ ((fstosnd+savV)*(totalConfidence+ConfDif+spdfirst) ) ,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ]))) 
}

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )


IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))
(plmodfstOV <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) )
pdf(paste0(curfigpath, "RPDbyOVControllingForConfFredSim",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(plmodfstOV)
dev.off()

if (doBias==1 ){
eff_df <- Effect(c("fstosnd", "cConfBias"), Choicemod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cConfBias <- as.factor(IA$cConfBias)

plmodfstbt<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= cConfBias)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = cConfBias),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1))
}


eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)

(plmodfst<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) )


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1)) )

IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)

(plmodsnd<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssubg)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) )

if (doBias==1){
pdf(paste0(curfigpath, "OVByConf",i,".pdf"), width = 12, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(multiplot(plmodfstbt, plmodfst, plmodsnd, cols=3))
dev.off()
}else{
pdf(paste0(curfigpath, "OVByConf",i,".pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(multiplot(plmodfst, plmodsnd, cols=2))
dev.off()
}
## 3 way w RPD
if (doBias==1){
eff_df <- Effect(c("fstosnd", "cConfBias", "spdfirst"), Choicemod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(IA$spdfirst)
IA$cConfBias <- as.factor(IA$cConfBias)

plmodfst<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative First Presentation Duration", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+ facet_wrap(~cConfBias, nrow =1)+
  scale_fill_manual(name="Relative First Presentation Duration", values=darkcolssub)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="bottom")+theme(strip.background = element_rect(colour="white",fill="white"))

pdf(paste0(curfigpath, "BiasPDSVD",i,".pdf"), width = 15, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(plmodfst)
dev.off()
}

############## by item ######################
# 
# if (doBias==1 ){
# print (summary(Choicemod0 <- glm(isfirstIchosen ~ (sfstItemV+ssndItemVal)* ((ConfDif+cConfBias*spdfirst)+totalConfidence)+cRT,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ])))  
# }else{
# print (summary(Choicemod0 <- glm(isfirstIchosen ~ (sfstItemV+ssndItemVal)* ((ConfDif+spdfirst)+totalConfidence)+cRT,family=binomial(link='logit') , data = a1c[a1c$risk_aversion_level==i, ])))  
# }
# 
# 
# eff_df <- Effect(c("sfstItemV", "ConfDif"), Choicemod0, xlevels=list(sfstItemV =seq(min(a1c$sfstItemV ), max(a1c$sfstItemV), 0.1)) )
# 
# #plot(eff_df)
# IA <- as.data.frame(eff_df)
# IA$ConfDif <- as.factor(IA$ConfDif)
# 
# IAf<- IA
# IAf$Item <- rep("first", length(IAf$fit))
# IAf <- plyr:::rename(IAf, c("sfstItemV" = "ItemV"))
# 
# eff_df <- Effect(c("ssndItemVal", "ConfDif"), Choicemod0, xlevels=list(ssndItemVal =seq(min(a1c$ssndItemVal ), max(a1c$ssndItemVal), 0.1)) )
# 
# IA <- as.data.frame(eff_df)
# IA$ConfDif <- as.factor(IA$ConfDif)
# IAs<- IA
# IAs$Item <- rep("second", length(IAs$fit))
# IAs <- plyr:::rename(IAs, c("ssndItemVal"="ItemV"))
# 
# IAall <- rbind(IAf, IAs)
# 
# plmodsndConfDif<- ggplot(data=IAall, aes(x=ItemV, y=fit , linetype= Item, color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IAall, aes(x=ItemV, max = fit + se, min = fit- se,  linetype= Item,fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
#   scale_fill_manual(name="Confidence\n Difference", values=darkcolssubg)+geom_line(data=IAall, aes(x=ItemV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right")
# 
# 
# pdf(paste0(curfigpath, "ByItemlftConfDiff",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(plmodsndConfDif)
# dev.off()
# 
# 
# 
# if (doBias==1){
# eff_df <- Effect(c("sfstItemV", "cConfBias"), Choicemod0, xlevels=list(sfstItemV =seq(min(a1c$sfstItemV ), max(a1c$sfstItemV), 0.1)) )
# 
# #plot(eff_df)
# IA <- as.data.frame(eff_df)
# IA$cConfBias <- as.factor(IA$cConfBias)
# 
# IAf<- IA
# IAf$Item <- rep("first", length(IAf$fit))
# IAf <- plyr:::rename(IAf, c("sfstItemV" = "ItemV"))
# 
# eff_df <- Effect(c("ssndItemVal", "cConfBias"), Choicemod0, xlevels=list(ssndItemVal =seq(min(a1c$ssndItemVal ), max(a1c$ssndItemVal), 0.1)) )
# 
# IA <- as.data.frame(eff_df)
# IA$cConfBias <- as.factor(IA$cConfBias)
# IAs<- IA
# IAs$Item <- rep("second", length(IAs$fit))
# IAs <- plyr:::rename(IAs, c("ssndItemVal" = "ItemV"))
# 
# IAall <- rbind(IAf, IAs)
# 
# plmodsndBias<- ggplot(data=IAall, aes(x=ItemV, y=fit , linetype= Item, color= cConfBias )) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IAall, aes(x=ItemV, max = fit + se, min = fit- se,  linetype= Item,fill = cConfBias),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
#   scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+geom_line(data=IAall, aes(x=ItemV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right")
# 
# 
# pdf(paste0(curfigpath, "ByItemlftConfBias",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(plmodsndBias)
# dev.off()
# 
# }
# 
# 
# eff_df <- Effect(c("sfstItemV", "totalConfidence"), Choicemod0, xlevels=list(sfstItemV =seq(min(a1c$sfstItemV ), max(a1c$sfstItemV), 0.1)) )
# 
# 
# IA <- as.data.frame(eff_df)
# IA$totalConfidence <- as.factor(IA$totalConfidence)
# 
# IAf<- IA
# IAf$Item <- rep("first", length(IAf$fit))
# IAf <- plyr:::rename(IAf, c("sfstItemV" = "ItemV"))
# 
# eff_df <- Effect(c("ssndItemVal", "totalConfidence"), Choicemod0, xlevels=list(ssndItemVal =seq(min(a1c$ssndItemVal ), max(a1c$ssndItemVal), 0.1)) )
# 
# IA <- as.data.frame(eff_df)
# IA$totalConfidence <- as.factor(IA$totalConfidence)
# IAs<- IA
# IAs$Item <- rep("second", length(IAs$fit))
# IAs <- plyr:::rename(IAs, c("ssndItemVal" = "ItemV"))
# 
# IAall <- rbind(IAf, IAs)
# 
# plmodsndTotalConf<- ggplot(data=IAall, aes(x=ItemV, y=fit , linetype= Item, color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IAall, aes(x=ItemV, max = fit + se, min = fit- se,  linetype= Item,fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
#   scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IAall, aes(x=ItemV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right")
# 
# 
# pdf(paste0(curfigpath, "ByItemTotalConf",i,".pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(plmodsndTotalConf)
# dev.off()
# 
# 
# if (doBias==1){
# pdf(paste0(curfigpath, "ByItemConfAll",i,".pdf"), width = 14, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(multiplot(plmodsndBias, plmodsndTotalConf, plmodsndConfDif, cols =3))
# dev.off()
# }else{
# pdf(paste0(curfigpath, "ByItemConfAll",i,".pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# print(multiplot( plmodsndTotalConf, plmodsndConfDif, cols =2))
# dev.off()
# }

## people with a higher confidence bias have shorter RTs

## check RT predictions for Fred

if (doBias==1){
print (summary(RTmod0 <- lm(log(RT*1000)~ (sVD+fstosnd +(savV)) *(spdfirst)+(totalConfidence+ConfDif+cConfBias), data =a1c)))
}else{
print (summary(RTmod0 <- lm(log(RT*1000)~ (sVD+fstosnd +(savV)) *(spdfirst)+(totalConfidence+ConfDif), data =a1c)))
}

eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(IA$spdfirst)

plmodfst <- ggplot(data=IA, aes(x=fstosnd, y=fit, color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se,  fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+#ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) +theme(strip.background = element_rect(colour="white",fill="white"))+ theme(legend.position="right") #c(0.8, 0.2)

pdf(paste0(curfigpath, "RTRVRPD",i,".pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfst)
dev.off()
# 
if (doBias==1){
eff_df <- Effect(c("cConfBias"), RTmod0 )

IA <- as.data.frame(eff_df)

plmodfstbtCB<- ggplot(data=IA, aes(x=cConfBias, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cConfBias, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ #ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Bias") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1))
}

eff_df <- Effect(c("ConfDif"), RTmod0 )

IA <- as.data.frame(eff_df)

plmodfstbtCD<- ggplot(data=IA, aes(x=ConfDif, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=ConfDif, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ #ylim(10.9, 11.2)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Difference") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("totalConfidence"), RTmod0 )

IA <- as.data.frame(eff_df)

plmodfstbtTC<- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ #ylim(10.9, 11.2)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 


if (doBias==1){
pdf(paste0(curfigpath, "RTConfAll",i,".pdf"), width = 12, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstbtCB,plmodfstbtTC, plmodfstbtCD,  cols =3)
dev.off()
}else{
pdf(paste0(curfigpath, "RTConfAll",i,".pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
print(multiplot(plmodfstbtTC, plmodfstbtCD,  cols =2))
dev.off()
}
# }

