
library(languageR)
library(lme4)
library(MASS)
library(ggplot2) #cookbook for R/ Graphs
library(memisc)
library(reshape)
library(reshape2) #melt and cast -> restructure and aggregate data
library(data.table)
library(psych)
library(doBy)
library(heplots)
library(plyr) #necessary for ddply
library(matrixStats) 
library(foreign) 
library(Hmisc)
library(lmerTest)
library (stringr)
library(gdata)
library(Rmisc)
library(effects)
library(RColorBrewer)
library(sjPlot)
library(buildmer)

# get to the right place:
fileLoc <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(fileLoc) # go to script location first
#setwd("../..") 

basepath <- getwd()

didLmerConverge = function(lmerModel){
  relativeMaxGradient=signif(max(abs(with(lmerModel@optinfo$derivs,solve(Hessian,gradient)))),3)
  if (relativeMaxGradient < 0.001) {
    cat(sprintf("\tThe relative maximum gradient of %s is less than our 0.001 criterion.\n\tYou can safely ignore any warnings about a claimed convergence failure.\n\n", relativeMaxGradient))
  }
  else {
    cat(sprintf("The relative maximum gradient of %s exceeds our 0.001 criterion.\nThis looks like a real convergence failure; maybe try simplifying your model?\n\n", relativeMaxGradient))
  }
}


cursimPath = '/model/results/2025-05-06/simulations/'

### load data ###
# choice data
input_file = paste0(basepath, '/data/allSubDataTablePilot.xls')
a1 = read.xls(input_file)
# Rating data
input_file =paste0(basepath, '/data/allSubRateDataTablePilot.xls')
aR = read.xls(input_file)
aR <- aR[!is.na(aR$RateRT1),]

## load simulation data
input_file =paste0(basepath, cursimPath, '/1-main.csv') #qualitative_sim_v3
aSIMb <-   read.csv(input_file)
aSIMb <- aSIMb[aSIMb$rt <5,]

input_file =paste0(basepath, cursimPath, '/1-flatprior.csv') #qualitative_sim_v3
aSIMbfp <-   read.csv(input_file)
aSIMbfp <- aSIMbfp[aSIMbfp$rt <5,]

input_file =paste0(basepath, cursimPath, '/1-zeroprior.csv') #qualitative_sim_v3
aSIMbzp <-   read.csv(input_file)
aSIMbzp <- aSIMbzp[aSIMbzp$rt <5,]

# Study 2
input_file =paste0(basepath, cursimPath, '/2-main.csv') #qualitative_sim_v3
aSIMc <-   read.csv(input_file)
aSIMc <- aSIMc[aSIMc$rt <5,]

input_file =paste0(basepath, cursimPath, '/2-nometa.csv') #qualitative_sim_v3
aSIMcac <-   read.csv(input_file)
aSIMcac <- aSIMcac[aSIMcac$rt <5,]


input_file =paste0(basepath, cursimPath, '/2-overconf.csv') #qualitative_sim_v3
aSIMcoc <-   read.csv(input_file)
aSIMcoc <- aSIMcoc[aSIMcoc$rt <5,]



### color definitions ###
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
## palette for relative presentation effects
darkcols <- brewer.pal(8, "RdGy")
darkcolssub <-  c("#C43C20", "#A35C89", "#775C89", "#4D5D8A", "#4E7CBE")#darkcols[c(1:2, 6:8)]
# Confidence palette
darkcolsg <- brewer.pal(8, "Greens")
darkcolssubg <-  darkcolsg[c(4:8)]
greys <- brewer.pal(8, "Greys")
greyssub <- greys[c(4:8)]

### exclude bad subjects and version 502###
str(a1)
a1 <- a1[!a1$Version == 502 & !a1$SubNum==1019 & !a1$SubNum==1051 & !a1$SubNum==1055,]

# convert variables and set contrasts
a1$SubNum <- as.factor(a1$SubNum)
a1$isERROR <- as.factor(a1$isERROR) # how many trials did they reverse their choice?
a1$isfirstIchosen <- as.numeric(a1$isFirstChosen) # main choice DV
a1$pdfirst <-  a1$relfstpresDur # relative first item presentation
a1$pdfirst <- (a1$fstItemPresDur/(a1$fstItemPresDur + a1$sndItemPresDur))
a1$sVD <- scale(a1$VD/10, scale=FALSE, center=TRUE) # value difference
a1$fstosnd <- (a1$fstItemV - a1$sndItemVal)/10
a1$savV <- scale(a1$ASV/10, scale=FALSE, center=TRUE) # overall value


aR$SubNum <- as.factor(aR$SubNum)
a1$subavV <- rep(NA_real_, length(a1$SubNum))
for (i in levels(a1$SubNum)){
  print(i)
  tmpmeanSubV <- mean(aR$Rating1[aR$SubNum==i], na.rm=TRUE)
  if (max(aR$Rating1[aR$SubNum==i])<=1){
    tmpmeanSubV = tmpmeanSubV*10
  }
  print(tmpmeanSubV)
  a1$subavV[a1$SubNum==i] <- a1$ASV[a1$SubNum==i] - tmpmeanSubV
}



### compute same vars for simulated data
# main model
aSIMb$SubNum <- factor(aSIMb$subject)
aSIMb$isfirstIchosen <- as.numeric(aSIMb$choice ==1)
aSIMb$sVD <- scale(abs(aSIMb$val1 -aSIMb$val2)/10, scale=FALSE, center=TRUE)
aSIMb$savV <- scale((aSIMb$val1+aSIMb$val2)/20, scale=FALSE, center=TRUE)
aSIMb$RT <- (aSIMb$pt1+aSIMb$pt2)
aSIMb$cRT <- scale(aSIMb$RT, scale=FALSE, center=TRUE)
aSIMb$sfstItemV <- scale(aSIMb$val1, scale=FALSE, center=TRUE)
aSIMb$ssndItemVal <- scale(aSIMb$val2, scale=FALSE, center=TRUE)
aSIMb$fstosnd <- (aSIMb$val1 -aSIMb$val2)/10
aSIMb$spdfirst<- (aSIMb$pt1/(aSIMb$pt1+aSIMb$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)

aSIMb$ASV <- (aSIMb$val1+aSIMb$val2)/2
aSIMb$isLongFirst <- aSIMb$initpresdur1>aSIMb$initpresdur2

pfirstlongSIMb <- ggplot(data=aSIMb[!is.na(aSIMb$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=aSIMb, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))


# flat prior model
aSIMbfp$SubNum <- factor(aSIMbfp$subject)
aSIMbfp$isfirstIchosen <- as.numeric(aSIMbfp$choice ==1)
aSIMbfp$sVD <- scale(abs(aSIMbfp$val1 -aSIMbfp$val2)/10, scale=FALSE, center=TRUE)
aSIMbfp$savV <- scale((aSIMbfp$val1+aSIMbfp$val2)/20, scale=FALSE, center=TRUE)
aSIMbfp$RT <- (aSIMbfp$pt1+aSIMbfp$pt2)
aSIMbfp$cRT <- scale(aSIMbfp$RT, scale=FALSE, center=TRUE)
aSIMbfp$sfstItemV <- scale(aSIMbfp$val1, scale=FALSE, center=TRUE)
aSIMbfp$ssndItemVal <- scale(aSIMbfp$val2, scale=FALSE, center=TRUE)
aSIMbfp$fstosnd <- (aSIMbfp$val1 -aSIMbfp$val2)/10
aSIMbfp$spdfirst<- (aSIMbfp$pt1/(aSIMbfp$pt1+aSIMbfp$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)


aSIMbfp$ASV <- (aSIMbfp$val1+aSIMbfp$val2)/2
aSIMbfp$isLongFirst <- aSIMbfp$initpresdur1>aSIMbfp$initpresdur2

pfirstlongSIMbfp <- ggplot(data=aSIMbfp[!is.na(aSIMbfp$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=aSIMbfp, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))

# zero prior model
aSIMbzp$SubNum <- factor(aSIMbzp$subject)
aSIMbzp$isfirstIchosen <- as.numeric(aSIMbzp$choice ==1)
aSIMbzp$sVD <- scale(abs(aSIMbzp$val1 -aSIMbzp$val2)/10, scale=FALSE, center=TRUE)
aSIMbzp$savV <- scale((aSIMbzp$val1+aSIMbzp$val2)/20, scale=FALSE, center=TRUE)
aSIMbzp$RT <- (aSIMbzp$pt1+aSIMbzp$pt2)
aSIMbzp$cRT <- scale(aSIMbzp$RT, scale=FALSE, center=TRUE)
aSIMbzp$sfstItemV <- scale(aSIMbzp$val1, scale=FALSE, center=TRUE)
aSIMbzp$ssndItemVal <- scale(aSIMbzp$val2, scale=FALSE, center=TRUE)
aSIMbzp$fstosnd <- (aSIMbzp$val1 -aSIMbzp$val2)/10
aSIMbzp$spdfirst<- (aSIMbzp$pt1/(aSIMbzp$pt1+aSIMbzp$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)


aSIMbzp$ASV <- (aSIMbzp$val1+aSIMbzp$val2)/2
aSIMbzp$isLongFirst <- aSIMbzp$initpresdur1> aSIMbzp$initpresdur2

pfirstlongSIMbzp <- ggplot(data=aSIMbzp[!is.na(aSIMbzp$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=aSIMbzp, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))

##### main choice analyses ####


# df |> 
#   drop_na(initpresdur2) |> 
#   mutate(longfirst = initpresdur1 > .311) |> 
#   ggplot(aes(val1 + val2, 1 * (choice == 1), color=factor(longfirst))) +
#   stat_summary_bin(fun.data=mean_cl_boot, bins=5) +
#   stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)
# ylab("choose first")

# generate raw data plot for attention effect on choice
## true attention manipulation variable
a1$isLongFirst <- a1$initpresdur1>a1$initpresdur2


#No fancy regressions or anything, and the color is a true independent variable—I feel like this is better

# subsetting data into separate versions
a1b <- a1[a1$Version==503,]
a1b$SubNum <- factor(a1b$SubNum)
a1b$sVD <- scale(a1b$VD/10, scale=FALSE, center=TRUE)
a1b$savV <- scale(a1b$ASV/10, scale=FALSE, center=TRUE)
a1b$cRT <- scale(a1b$RT, scale=FALSE, center=TRUE)
a1b$sfstItemV <- scale(a1b$fstItemV, scale=FALSE, center=TRUE)
a1b$ssndItemVal <- scale(a1b$sndItemVal, scale=FALSE, center=TRUE)
a1b$spdfirst<- a1b$pdfirst - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)
a1b$relpresdur <- a1b$fstItemPresDur - a1b$sndItemPresDur
a1b$ssubavV <- scale(a1b$subavV, scale=FALSE, center=TRUE)


a1c <- a1[a1$Version==504,]
a1c$SubNum <- factor(a1c$SubNum)
a1c$sVD <- scale(a1c$VD/10, scale=FALSE, center=TRUE)
a1c$savV <- scale(a1c$ASV/10, scale=FALSE, center=TRUE)
a1c$cRT <- scale(a1c$RT, scale=FALSE, center=TRUE)
a1c$sfstItemV <- scale(a1c$fstItemV, scale=FALSE, center=TRUE)
a1c$ssndItemVal <- scale(a1c$sndItemVal, scale=FALSE, center=TRUE)
a1c$spdfirst<- a1c$pdfirst - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)


for (i in levels(a1c$SubNum) ) 
{print(i)
  a1c$ConfBias[a1c$SubNum==i]  <-  mean(c(a1c$fstConfidence[a1c$SubNum==i], a1c$sndConfidence[a1c$SubNum==i]))
}


a1c$cConfBias <- scale(a1c$ConfBias, scale=FALSE, center= TRUE)
a1c$sumConfidence <- a1c$fstConfidence+a1c$sndConfidence
a1c$totalConfidence <- (a1c$sumConfidence)/2 - a1c$ConfBias # total confidence is demeaned
#a1c$totalConfidence <- scale(a1c$sumConfidence, scale=FALSE, center = TRUE)
a1c$cfstConfidence <- scale(a1c$fstConfidence, scale=FALSE, center = TRUE)
a1c$csndConfidence <- scale(a1c$sndConfidence, scale=FALSE, center = TRUE)
a1c$ConfDif <- a1c$fstConfidence-a1c$sndConfidence# don't center





## plot this for study 1 and 2

pfirstlongS1 <- ggplot(data=a1b[!is.na(a1b$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=a1, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))
multiplot(pfirstlongSIMb, pfirstlongS1, cols = 2)

pfirstlongS2 <- ggplot(data=a1c[!is.na(a1c$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=a1, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))

### compute same vars for simulated data

# main model
aSIMc$SubNum <- factor(aSIMc$subject)
aSIMc$isfirstIchosen <- as.numeric(aSIMc$choice ==1)
aSIMc$sVD <- scale(abs(aSIMc$val1 -aSIMc$val2)/10, scale=FALSE, center=TRUE)
aSIMc$savV <- scale((aSIMc$val1+aSIMc$val2)/20, scale=FALSE, center=TRUE)

aSIMc$RT <- (aSIMc$pt1+aSIMc$pt2)
aSIMc$cRT <- scale(aSIMc$RT, scale=FALSE, center=TRUE)
aSIMc$sfstItemV <- scale(aSIMc$val1, scale=FALSE, center=TRUE)
aSIMc$ssndItemVal <- scale(aSIMc$val2, scale=FALSE, center=TRUE)
aSIMc$fstosnd <- (aSIMc$val1 -aSIMc$val2)/10
aSIMc$spdfirst<- (aSIMc$pt1/(aSIMc$pt1+aSIMc$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)

for (i in levels(aSIMc$SubNum) ) 
{print(i)
  aSIMc$ConfBias[aSIMc$SubNum==i]  <-  mean(c(aSIMc$conf1[aSIMc$SubNum==i], aSIMc$conf2[aSIMc$SubNum==i]))
}

aSIMc$cConfBias <- scale(aSIMc$ConfBias, scale=FALSE, center= TRUE)
aSIMc$sumConfidence <- aSIMc$conf1+aSIMc$conf2
aSIMc$totalConfidence <- (aSIMc$sumConfidence)/2 - aSIMc$ConfBias
aSIMc$cfstConfidence <- scale(aSIMc$conf1, scale=FALSE, center = TRUE)
aSIMc$csndConfidence <- scale(aSIMc$conf2, scale=FALSE, center = TRUE)
aSIMc$ConfDif <- aSIMc$conf1-aSIMc$conf2# don't center

aSIMc$ASV <- (aSIMc$val1+aSIMc$val2)/2
aSIMc$isLongFirst <- aSIMc$initpresdur1>aSIMc$initpresdur2

pfirstlongSIMc <- ggplot(data=aSIMc[!is.na(aSIMc$isLongFirst),], aes(x=ASV, y=isfirstIchosen,group = isLongFirst, color = isLongFirst, fill=isLongFirst))+stat_smooth(method="glm", method.args = list(family="binomial"), alpha=0.2)+  stat_summary_bin(fun.data=mean_cl_boot, bins=5)+#geom_point(size = 5, position=position_dodge(.1))+#+ scale_fill_manual(name = "Reward level", values=c("#343085" ,"#F1E923"))+
  theme_classic()+xlab("Overall Value") + ylab("P(First Chosen)") +scale_colour_manual(name="first long", values=darkcolssub[c(1,5)])+scale_fill_manual(name="first long", values=darkcolssub[c(1,5)])+
  geom_line(data=a1, aes(x=ASV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=5, linetype=2, size=0.2)  + ylab("P(first chosen)") + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1))



# average confidence /AC model
aSIMcac$SubNum <- factor(aSIMcac$subject)
aSIMcac$isfirstIchosen <- as.numeric(aSIMcac$choice ==1)
aSIMcac$sVD <- scale(abs(aSIMcac$val1 -aSIMcac$val2)/10, scale=FALSE, center=TRUE)
aSIMcac$savV <- scale((aSIMcac$val1+aSIMcac$val2)/20, scale=FALSE, center=TRUE)
aSIMcac$RT <- (aSIMcac$pt1+aSIMcac$pt2)
aSIMcac$cRT <- scale(aSIMcac$RT, scale=FALSE, center=TRUE)
aSIMcac$sfstItemV <- scale(aSIMcac$val1, scale=FALSE, center=TRUE)
aSIMcac$ssndItemVal <- scale(aSIMcac$val2, scale=FALSE, center=TRUE)
aSIMcac$fstosnd <- (aSIMcac$val1 -aSIMcac$val2)/10
aSIMcac$spdfirst<- (aSIMcac$pt1/(aSIMcac$pt1+aSIMcac$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)

for (i in levels(aSIMcac$SubNum) ) 
{print(i)
  aSIMcac$ConfBias[aSIMcac$SubNum==i]  <-  mean(c(aSIMcac$conf1[aSIMcac$SubNum==i], aSIMcac$conf2[aSIMcac$SubNum==i]))
}

aSIMcac$cConfBias <- scale(aSIMcac$ConfBias, scale=FALSE, center= TRUE)
aSIMcac$sumConfidence <- aSIMcac$conf1+aSIMcac$conf2
aSIMcac$totalConfidence <- (aSIMcac$sumConfidence)/2 - aSIMcac$ConfBias
aSIMcac$cfstConfidence <- scale(aSIMcac$conf1, scale=FALSE, center = TRUE)
aSIMcac$csndConfidence <- scale(aSIMcac$conf2, scale=FALSE, center = TRUE)
aSIMcac$ConfDif <- aSIMcac$conf1-aSIMcac$conf2# don't center


# overconfidence model/oc model 
aSIMcoc$SubNum <- factor(aSIMcoc$subject)
aSIMcoc$isfirstIchosen <- as.numeric(aSIMcoc$choice ==1)
aSIMcoc$sVD <- scale(abs(aSIMcoc$val1 -aSIMcoc$val2)/10, scale=FALSE, center=TRUE)
aSIMcoc$savV <- scale((aSIMcoc$val1+aSIMcoc$val2)/20, scale=FALSE, center=TRUE)
aSIMcoc$RT <- (aSIMcoc$pt1+aSIMcoc$pt2)
aSIMcoc$cRT <- scale(aSIMcoc$RT, scale=FALSE, center=TRUE)
aSIMcoc$sfstItemV <- scale(aSIMcoc$val1, scale=FALSE, center=TRUE)
aSIMcoc$ssndItemVal <- scale(aSIMcoc$val2, scale=FALSE, center=TRUE)
aSIMcoc$fstosnd <- (aSIMcoc$val1 -aSIMcoc$val2)/10
aSIMcoc$spdfirst<- (aSIMcoc$pt1/(aSIMcoc$pt1+aSIMcoc$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)

for (i in levels(aSIMcoc$SubNum) ) 
{print(i)
  aSIMcoc$ConfBias[aSIMcoc$SubNum==i]  <-  mean(c(aSIMcoc$conf1[aSIMcoc$SubNum==i], aSIMcoc$conf2[aSIMcoc$SubNum==i]))
}

#aSIMcoc$cConfBias <- scale(aSIMcoc$ConfBias, scale=FALSE, center= TRUE)
aSIMcoc$sumConfidence <- aSIMcoc$conf1+aSIMcoc$conf2
aSIMcoc$totalConfidence <- (aSIMcoc$sumConfidence)/2 - aSIMcoc$ConfBias
aSIMcoc$cfstConfidence <- scale(aSIMcoc$conf1, scale=FALSE, center = TRUE)
aSIMcoc$csndConfidence <- scale(aSIMcoc$conf2, scale=FALSE, center = TRUE)
aSIMcoc$ConfDif <- aSIMcoc$conf1-aSIMcoc$conf2# don't center


aSIMcoc$cConfBias <- scale(aSIMcoc$subjective_offset, center=TRUE, scale=FALSE)

## data descriptions

nTrials <- ddply(a1b, .(SubNum), summarise,
                 Trials    = max(Trial),
                 subisError = mean(as.numeric(as.character(isERROR)), na.rm=TRUE))

median(nTrials$Trials)
min(nTrials$Trials)
sd(nTrials$Trials)

mean(nTrials$subisError)

nTrials <- ddply(a1c, .(SubNum), summarise,
                 Trials    = max(Trial),
                 subisError = mean(as.numeric(as.character(isERROR)), na.rm=TRUE))

median(nTrials$Trials)
min(nTrials$Trials)
sd(nTrials$Trials)

mean(nTrials$subisError)

############# Study 1 controlling attention ##############

a1b <- a1b[!a1b$Choice==-1,]

# 1) How do values and presentation duration affect choice?

# we don't let the model fit a random intercept because that won't converge. There's no variance in bias.
# print (summary(Choicemod0 <- buildmer(isfirstIchosen ~ (fstosnd + savV)* spdfirst+cRT  +(0+(fstosnd + savV)* spdfirst+cRT |SubNum), data = a1b, 
#                                    family = binomial))) 

## this is the  model we get from buildmer. That doesn't allow us to test the interaction of presentation duration with relative value, hence we run the one below 
print (summary(Choicemod0 <- glmer(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV +      (0 + fstosnd + spdfirst | SubNum), data = a1b[!a1b$numcycles==1,], 
                                   family = binomial))) 

sv1_max <- svd(getME(Choicemod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(Choicemod0, transform=NULL)

test1 <- summary(Choicemod0) # saving out coefficients for sensitivity analysis


## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0, xlevels=list(fstosnd =seq(min(a1b$fstosnd, na.rm=TRUE ), max(a1b$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IARV <- IA
pRVS1<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) 


## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0 )

IA <- as.data.frame(eff_df)
pSPDS1<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))


## ... but the the relationship between presentation duration and choice depends on value:
# participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
eff_df <- Effect(c("savV", "spdfirst"), Choicemod0, xlevels=list(savV =seq(min(a1b$savV ), max(a1b$savV), 0.1), spdfirst =seq(min(a1b$spdfirst, na.rm=TRUE ), max(a1b$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVS1 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.2, 0.25)) +#
  coord_cartesian(ylim = c(0, 1)) 


### test interaction differently
a1b$isGreaterZero <- as.factor(a1b$savV>0)
contrasts(a1b$isGreaterZero) <- contr.sdif(2)
print (summary(Choicemod0S1IA <- glmer(isfirstIchosen ~ fstosnd + cRT +  isGreaterZero*( spdfirst) +      (1 + fstosnd | SubNum), data = a1b[!a1b$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                       family = binomial))) 

tab_model(Choicemod0S1IA, transform =NULL)

# significant positive effect of relative fixation duration above the mean. Significant negative estimate for relative presentation duration below the mean.
print (summary(Choicemod0S1IAn <- glmer(isfirstIchosen ~ fstosnd + cRT + isGreaterZero/( spdfirst) +      (1 + fstosnd | SubNum), data = a1b[!a1b$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                        family = binomial))) 
tab_model(Choicemod0S1IAn, transform =NULL)

## How about response times?

# print (summary(RTmod0 <- buildmer(log(RT*1000)~ (sVD+savV+fstosnd)*spdfirst+((sVD+savV+fstosnd)*spdfirst|SubNum), a1b[!a1b$Choice==-1,], 
#                               REML=FALSE)))


print (summary(RTmod0 <- lmer(log(RT*1000)~  sVD + savV  +  
                                (1 + sVD + savV | SubNum), a1b[!a1b$Choice==-1,], 
                              REML=FALSE)))

sv1_max <- svd(getME(RTmod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(RTmod0)

tab_model(Choicemod0,RTmod0, transform=NULL, show.stat = TRUE, col.order= c("est", "ci", "stat", "p"))

eff_df <- effect("sVD", RTmod0,  xlevels=list(sVD=seq(min(a1b$sVD, na.rm=TRUE), max(a1b$sVD, na.rm=TRUE), 0.1)))
contmain <- as.data.frame(eff_df)
contmain$sVD <- contmain$sVD - min(contmain$sVD, na.rm=TRUE)

contmainC <- contmain
contmainC$ValType <- rep("VD", length(contmain$fit))
contmainC <- rename(contmainC, c("Value"="sVD"))

eff_df <- effect("savV", RTmod0,  xlevels=list(savV=seq(min(a1b$savV, na.rm=TRUE), max(a1b$savV, na.rm=TRUE), 0.1)))
contmain <- as.data.frame(eff_df)
contmain$savV <- contmain$savV - min(contmain$savV, na.rm=TRUE)

contmainuC <- contmain
contmainuC$ValType <- rep("OV", length(contmain$fit))
contmainuC <- rename(contmainuC, c("Value"="savV"))

contmainall <- rbind(contmainC, contmainuC)

contmainall$ValType <- ordered(contmainall$ValType, levels=c("VD", "OV"))

contmainallS1 <- contmainall

pRTS1 <- ggplot(data=contmainall, aes(x=Value, y=fit, color = ValType, linetype= ValType))+theme_bw(12)+ geom_ribbon(data=contmainall, aes(x=Value, max = upper, min = lower, fill = ValType),alpha=0.2, inherit.aes = FALSE) + geom_line()+ylim(7.3, 7.8)+
  scale_linetype_manual(values = c(1,1)) +xlab("Value") + ylab("RT")+scale_color_manual(values=c("#000000","#C0C0C0C0"))+scale_fill_manual(values=c("#000000","#C0C0C0C0")) + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.8, 0.8))



### Now we do that same thing for Version 504

a1c <- a1c[!a1c$Choice==-1,]

## ok, cool. So, how about confidence?

# How does Confidence affect choices?
## before looking into that, let's generate some plots.

# we want a q-plot of item 1 vs 2 confidence

pConfOverview1<- ggplot(data=a1c, aes(x= sumConfidence, y=ConfDif)) + geom_point()+
  geom_count() + scale_size_area()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssubg) +theme_classic(12)+ #ylim(0, 1)+
  xlab("Overall Confidence") + ylab("Relative Confidence") + theme(legend.position=c(0.25, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(-5, 5)) 

pConfOverview2<- ggplot(data=a1c, aes(x=fstConfidence, y=sndConfidence)) + geom_point() +theme_classic(12)+geom_count() + scale_size_area()+ #ylim(0, 1)+
  xlab("First Item Confidence") + ylab("Second Item Confidence") + theme(legend.position=c(0.2, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 6), xlim = c(0,6)) 

library(ggExtra)
pdf(paste0(basepath, "/Figures/ConfidenceOvR.pdf"), width = 4, height = 4)
ggMarginal(pConfOverview1, type = "histogram")
dev.off()
pdf(paste0(basepath, "/Figures/Confidence1stv2nd.pdf"), width = 4, height = 4)
ggMarginal(pConfOverview2, type = "histogram")
dev.off()

aRc <- aR[aR$Version==504 & aR$isInChoiceSet ==1,]

aRc$cConfidence <- scale(aRc$Confidence, scale=FALSE, center = TRUE)
print (summary(modVConf <- lmer(Confidence~ Rating1+ I(Rating1*Rating1) +(Rating1|SubNum), aRc, 
                                REML=FALSE)))

sv1_max <- svd(getME(modVConf, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

didLmerConverge(modVConf)

tab_model(modVConf)

eff_df <- Effect(c("Rating1"), modVConf, xlevels=list(Rating1 =seq(min(aRc$Rating1, na.rm=TRUE ), max(aRc$Rating1, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pConfV<- ggplot(data=IA, aes(x=Rating1, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=Rating1, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  geom_vline(xintercept=5, linetype=2, size=0.2) + xlab("Item Rating") + ylab("Confidence Rating") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))#+
#coord_cartesian(ylim = c(0, 1)) 
pdf(paste0(basepath, "/Figures/ConfidencebValue.pdf"), width = 4, height = 4)
pConfV
dev.off()

## checking Rating Consistency across pre- and post choice artings
aRc$RateVar <- abs(aRc$RateConsistency)

subRate <- data.frame(Rating1 = c(mean(aRc$Rating1, na.rm=TRUE),sd(aRc$Rating1, na.rm=TRUE)),
                      Rating2 = c(mean(aRc$Rating2, na.rm=TRUE),sd(aRc$Rating2, na.rm=TRUE)),
                      RateVar = c(mean(aRc$RateVar, na.rm=TRUE), sd(aRc$RateVar, na.rm=TRUE) ))
row.names(subRate) <- c("Mean", "SD")
colnames(subRate) <- c("Rating 1", "Rating 2", "Rating Var")
library(dplyr)
library(htmlTable)
subRate%>%
  mutate_if(is.numeric, round, digits =2)%>%
  htmlTable
cor.test(aRc$Rating1, aRc$Rating2)

print (summary(modConfCons <- lmer(RateVar~ cConfidence+Rating1+ I(Rating1*Rating1)  +(cConfidence+Rating1|SubNum), aRc, 
                                   REML=FALSE)))

didLmerConverge(modConfCons)

tab_model(modConfCons, show.stat = TRUE, show.df = TRUE, col.order= c("est", "ci", "stat", "df.error", "p"))

eff_df <- Effect(c("cConfidence"), modConfCons, xlevels=list(Rating1 =seq(min(aRc$cConfidence, na.rm=TRUE ), max(aRc$cConfidence, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pConfCons<- ggplot(data=IA, aes(x=cConfidence, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  xlab("Value Confidence (Rating 1)") + ylab("Rating Deviation") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))#+


## original model

# 
# print (summary(Choicemod0 <- buildmer(isfirstIchosen ~ ((fstosnd +savV)* (totalConfidence+ConfDif+spdfirst))+cRT +(((fstosnd +savV)* (totalConfidence+ConfDif+spdfirst))+cRT |SubNum), data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
#                                    family = binomial))) 

#isfirstIchosen ~ 1 + fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum)

print (summary(Choicemod0S2 <- glmer(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum), data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                     family = binomial))) 

sv1_max <- svd(getME(Choicemod0S2, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  
tab_model(Choicemod0S2, transform =NULL)

## saving out coefficients for sensitivity analysis
test <- summary(Choicemod0S2)
test1sub <- test1$coefficients
testS2sub <- test$coefficients

write.csv(test1sub, "Study1Coefficients.csv")
write.csv(testS2sub, "Study2Coefficients.csv")

## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0S2, xlevels=list(fstosnd =seq(min(a1c$fstosnd, na.rm=TRUE ), max(a1c$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)

# plot S2 data on top of S1 data with dashed lines
pRV <- pRVS1 + geom_line(data=IA, aes(x=fstosnd, y=fit  ), linetype = 2, size = 0.5) + geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)

## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0S2 )

IA <- as.data.frame(eff_df)
pSPDS2 <- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0S2, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1), spdfirst =seq(min(a1c$spdfirst, na.rm = TRUE ), max(a1c$spdfirst, na.rm = TRUE), 0.2)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))
plmodfstOVS2 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line(linetype = 2, size = 0.5)+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="none") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0,1)) 


eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0S2, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)
pSetConfS2<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0S2, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1)) )


IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)
pConfDifS2<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


### test interaction differently
a1c$isGreaterZero <- as.factor(a1c$savV>0)
contrasts(a1c$isGreaterZero) <- contr.sdif(2)
print (summary(Choicemod0S2IA <- glmer(isfirstIchosen ~ fstosnd + cRT + totalConfidence +  fstosnd:totalConfidence +  isGreaterZero*( ConfDif + spdfirst) +      (1 + fstosnd | SubNum), data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                       family = binomial))) 

tab_model(Choicemod0S2IA, transform =NULL)

# significant positive effects of relative confidence and relative fixation duration above the mean. Significant negative estimate for relative confidence below the mean. Non significant negative effect of fixation duration below.
print (summary(Choicemod0S2IAn <- glmer(isfirstIchosen ~ fstosnd + cRT + totalConfidence +  fstosnd:totalConfidence +  isGreaterZero/( ConfDif + spdfirst) +      (1 + fstosnd | SubNum), data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                        family = binomial))) 
tab_model(Choicemod0S2IAn, transform =NULL)

## how about RT?
#a1c$logRT <- log(a1c$RT*1000)

# 
# print (summary(RTmod0 <- buildmer(log(RT*1000)~  sVD+savV+fstosnd*(spdfirst)+(totalConfidence+ConfDif)+(sVD+savV+fstosnd*(spdfirst)+(totalConfidence+ConfDif)|SubNum), a1c[!a1c$numcycles==1,], 
#                               REML=FALSE)))
#+ fstosnd + spdfirst + fstosnd:spdfirst


print (summary(RTmod0S2 <- lmer(log(RT*1000)~   sVD + savV + ConfDif + totalConfidence +(savV+sVD|SubNum), a1c[!a1c$numcycles==1,],
                                REML=FALSE)))

sv1_max <- svd(getME(RTmod0S2, "Tlist")[[1]])
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)

tab_model(RTmod0S2)

tab_model(Choicemod0S2,RTmod0S2, transform =NULL, show.stat = TRUE, col.order= c("est", "ci", "stat", "p"))


eff_df <- effect("sVD", RTmod0S2,  xlevels=list(sVD=seq(min(a1c$sVD, na.rm=TRUE), max(a1c$sVD, na.rm=TRUE), 0.1)))
contmain <- as.data.frame(eff_df)
contmain$sVD <- contmain$sVD - min(contmain$sVD, na.rm=TRUE)
pVDRT <- ggplot(data=contmain, aes(x=sVD, y=fit)) + geom_line(color="#000000")+theme_bw(12)+ geom_ribbon(data=contmain, aes(x=sVD, max = upper, min = lower),alpha=0.3, inherit.aes = FALSE)+
  xlab("VD") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="bottom")

contmainC <- contmain
contmainC$ValType <- rep("VD", length(contmain$fit))
contmainC <- rename(contmainC, c("Value"="sVD"))

eff_df <- effect("savV", RTmod0S2,  xlevels=list(savV=seq(min(a1c$savV, na.rm=TRUE), max(a1c$savV, na.rm=TRUE), 0.1)))
contmain <- as.data.frame(eff_df)
contmain$savV <- contmain$savV - min(contmain$savV, na.rm=TRUE)

contmainuC <- contmain
contmainuC$ValType <- rep("OV", length(contmain$fit))
contmainuC <- rename(contmainuC, c("Value"="savV"))

contmainall <- rbind(contmainC, contmainuC)

contmainall$ValType <- ordered(contmainall$ValType, levels=c("VD", "OV"))

contmainallS1$Study <- rep("Study1", length(contmainallS1$Value))
contmainall$Study <- rep("Study2", length(contmainall$Value))

contmainallS12 <- rbind(contmainallS1, contmainall)

pRT <- ggplot(data=contmainallS12, aes(x=Value, y=fit, color = interaction(Study,ValType), linetype=interaction(Study, ValType)))+theme_bw(12)+ geom_ribbon(data=contmainallS12, aes(x=Value, max = upper, min = lower, fill = interaction(Study,ValType)),alpha=0.2, inherit.aes = FALSE) + geom_line()+ylim(7.3, 7.8)+
  xlab("Value") + ylab("RT")+ scale_linetype_manual(name = "",values = c(1,2,1,2))+scale_color_manual(name = "",values=c("#000000","#000000","#77787B","#77787B"))+scale_fill_manual(name = "",values=c("#000000","#000000","#C0C0C0C0", "#C0C0C0C0")) + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.8, 0.8))

## here make figure for both studies simple behavior
pdf(paste0(basepath, "/Figures/Fig2B.pdf"), width = 4, height = 8)
multiplot(pRV,pRT)
dev.off()


eff_df <- Effect(c("ConfDif"), RTmod0S2 )

IA <- as.data.frame(eff_df)
plmodfstbtCDS2 <- ggplot(data=IA, aes(x=ConfDif, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=ConfDif, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+#ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Difference") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("totalConfidence"), RTmod0S2 )

IA <- as.data.frame(eff_df)
plmodfstbtTCS2 <- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+#ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

##################### Bias analyses #######################
#fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum)

print (summary(Choicemod0wb <- glmer(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + cConfBias + savV + fstosnd:cConfBias + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum)
                                     , data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                     family = binomial))) 

sv1_max <- svd(getME(Choicemod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(Choicemod0wb, transform =NULL)


print (summary(RTmod0wb <- lmer(log(RT*1000)~  sVD + savV + ConfDif + totalConfidence  +cConfBias +(savV+sVD|SubNum) , a1c[!a1c$numcycles==1,], 
                                REML=FALSE)))

sv1_max <- svd(getME(RTmod0wb, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(RTmod0wb)

tab_model(Choicemod0wb,RTmod0wb, transform =NULL, show.stat = TRUE, col.order= c("est", "ci", "stat", "p"))

eff_df <- Effect(c("cConfBias"), RTmod0wb )

IA <- as.data.frame(eff_df)
plmodfstbtCB<- ggplot(data=IA, aes(x=cConfBias, y=fit)) + geom_line() +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cConfBias, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+
  xlab("Confidence Bias") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("fstosnd", "cConfBias"), Choicemod0wb, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cConfBias <- as.factor(IA$cConfBias)
plmodfstbt<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= cConfBias)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=greyssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = cConfBias),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Bias", values=greyssub)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1))


############## generate "raw data" points for data plots #######

# 1) run regression with fstosnd only

print (summary(Choicemod0res <- glmer(isfirstIchosen ~ 0+fstosnd +      (0 + fstosnd + spdfirst | SubNum), data = a1b[!a1b$numcycles==1,], 
                                      family = binomial))) 

sv1_max <- svd(getME(Choicemod0res, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(Choicemod0res, transform=NULL)

print (summary(Choicemod0res2 <- glmer(isfirstIchosen ~ 0+fstosnd +      (0 + fstosnd + spdfirst | SubNum), data = a1c[!a1c$numcycles==1,], 
                                       family = binomial))) 

sv1_max <- svd(getME(Choicemod0res2, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

tab_model(Choicemod0res2, transform=NULL)

## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0res, xlevels=list(fstosnd =seq(min(a1b$fstosnd, na.rm=TRUE ), max(a1b$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IARV <- IA

eff_df <- Effect(c("fstosnd"), Choicemod0res2, xlevels=list(fstosnd =seq(min(a1b$fstosnd, na.rm=TRUE ), max(a1b$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IARV2 <- IA

# we have the predicted effect of RV. We will now subtract that for each 

a1b$RVpl <- round(a1b$fstosnd,1)
a1b$resC <- rep(NA_real_, length(a1b$RVpl))
for (ii in IARV$fstosnd){
  a1b$resC[a1b$RVpl ==ii] <- a1b$isfirstIchosen[a1b$RVpl ==ii] - IARV$fit[IARV$fstosnd==ii]
  
}

a1b$RPDpl <- round(a1b$spdfirst/2,1)
a1b$RPDpl  <- cut(a1b$spdfirst,
                  quantile(a1b$spdfirst, seq(0, 1, 1/5), na.rm=TRUE), include.lowest=T)#,
# labels=c("low","medium","high"))

a1b$OVpl <- round(a1b$ASV/2.5)

# same study 2
a1c$RVpl <- round(a1c$fstosnd,1)
a1c$resC <- rep(NA_real_, length(a1c$RVpl))
for (ii in IARV2$fstosnd){
  a1c$resC[a1c$RVpl ==ii] <- a1c$isfirstIchosen[a1c$RVpl ==ii] - IARV2$fit[IARV2$fstosnd==ii]
  
}

a1c$RPDpl <- round(a1c$spdfirst/2,1)
a1c$RPDpl  <- cut(a1c$spdfirst,
                  quantile(a1c$spdfirst, seq(0, 1, 1/4), na.rm=TRUE), include.lowest=T)#,
# labels=c("low","medium","high"))

a1c$OVpl <- round(a1c$ASV/2.5)

#sum1 <-summarySEwithin(a1b[! a1b$numcycles==1,], measurevar="isfirstIchosen", withinvars=c("RPDpl", "OVpl" ),  idvar="SubNum", na.rm=TRUE)
sum1 <-summarySE(a1b[! a1b$numcycles==1,], measurevar="resC", groupvars =c("RPDpl", "OVpl" ), na.rm=TRUE)
sum1$OVpl <- as.numeric(as.character(sum1$OVpl))*2.5
sum1$OVpl <- (sum1$OVpl-mean(a1b$ASV))/10
sum1$RPDpl <- as.factor(sum1$RPDpl)
levels(sum1$RPDpl) <-  levels(plmodfstOVS1$data$spdfirst)
sum1$spdfirst <- sum1$RPDpl
sum1$resC <- sum1$resC+0.5
sum1$fit <-sum1$resC

plmodfstOVS1wr <- plmodfstOVS1+ geom_point(data=sum1, aes(x=OVpl, y=fit, color= spdfirst )) +geom_line(data=sum1, aes(x=OVpl, y=fit, color= spdfirst ), linetype = 2, size = 0.5) +geom_errorbar(data=sum1, aes(x=OVpl, max = fit + se, min = fit- se, color = spdfirst), width = 0.01)


# also generate relative presentation duration plot
sum1 <-summarySE(a1b[! a1b$numcycles==1,], measurevar="resC", groupvars =c("RPDpl"), na.rm=TRUE)
sum1$spdfirst <- sum1$RPDpl
levels(sum1$RPDpl) <-  levels(plmodfstOVS1$data$spdfirst)
sum1$spdfirst <- as.numeric(as.character(sum1$RPDpl))
sum1$resC <- sum1$resC+0.5
sum1$fit <-sum1$resC

sum2 <-summarySE(a1c[! a1b$numcycles==1,], measurevar="resC", groupvars =c("RPDpl"), na.rm=TRUE)
sum2$spdfirst <- sum2$RPDpl
levels(sum2$RPDpl) <-  levels(plmodfstOVS2$data$spdfirst)
sum2$spdfirst <- as.numeric(as.character(sum2$RPDpl))
sum2$resC <- sum2$resC+0.5
sum2$fit <-sum2$resC


pSPDS1wd <- pSPDS1 + geom_point(data=sum1, aes(x=spdfirst, y=fit )) +geom_line(data=sum1, aes(x=spdfirst, y=fit), linetype = 2, size = 0.5, color = "grey") +geom_errorbar(data=sum1, aes(x=spdfirst, max = fit + se, min = fit- se), width = 0.01)


pSPDS2wd <- pSPDS2 +  geom_point(data=sum2, aes(x=spdfirst, y=fit )) +geom_line(data=sum2, aes(x=spdfirst, y=fit), linetype = 3, size = 0.5, color = "grey") +geom_errorbar(data=sum2, aes(x=spdfirst, max = fit + se, min = fit- se), width = 0.01)




## datapoints for inset Fig 3, save inset
sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="resC", groupvars =c("RPDpl", "OVpl" ), na.rm=TRUE)
sum1$OVpl <- as.numeric(as.character(sum1$OVpl))*2.5
sum1$OVpl <- (sum1$OVpl-mean(a1c$ASV))/10
sum1$RPDpl <- as.factor(sum1$RPDpl)
levels(sum1$RPDpl) <-  levels(plmodfstOVS2$data$spdfirst)
sum1$spdfirst <- sum1$RPDpl
sum1$resC <- sum1$resC+0.5
sum1$fit <-sum1$resC

plmodfstOVS2wr <- plmodfstOVS2+ geom_point(data=sum1, aes(x=OVpl, y=fit, color= spdfirst )) +geom_line(data=sum1, aes(x=OVpl, y=fit, color= spdfirst ), linetype = 3, size = 0.25) +geom_errorbar(data=sum1, aes(x=OVpl, max = fit + se, min = fit- se, color = spdfirst), width = 0.01)

pdf(paste0(basepath, "/Figures/Combi/Fig3_inset.pdf"), width = 4, height = 3)#, units = 'cm', res = 200, compression = 'lzw'
plmodfstOVS2wr
dev.off()


## goal: Fig 4 

# Total confidence
#pSetConfS2, plmodfstbtTCS2, pConfDifS2

a1c$totalConfpl <- round(a1c$totalConfidence)
a1c$totalConfpl[a1c$totalConfpl==-3] <- -4 # lumping -3 and -4 together to match the model prediction levels
a1c$RVpl2 <- ((round((a1c$RVpl*10)/5))*5)/10
sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="isfirstIchosen", groupvars =c("totalConfpl", "RVpl2" ), na.rm=TRUE)
sum1$RVpl2 <- as.numeric(as.character(sum1$RVpl2))
#sum1$RVpl <- (sum1$OVpl-mean(a1c$ASV))/10
sum1$fstosnd <- sum1$RVpl2
sum1$totalConfpl <- as.factor(sum1$totalConfpl)
levels(sum1$totalConfpl) <- levels(pSetConfS2$data$totalConfidence)
sum1 <- sum1[!sum1$N<10,]
sum1$totalConfidence <- sum1$totalConfpl
sum1$fit <-sum1$isfirstIchosen

pSetConfS2wr <- pSetConfS2+ geom_point(data=sum1, aes(x=fstosnd, y=fit, color= totalConfidence )) +geom_line(data=sum1, aes(x=fstosnd, y=fit, color= totalConfidence ), linetype = 2, size = 0.25) +geom_errorbar(data=sum1, aes(x=fstosnd, max = fit + se, min = fit- se, color = totalConfidence), width = 0.01)

## rt

print (summary(RTmod0S2res <- lmer(log(RT*1000)~   sVD + savV+(savV+sVD|SubNum), a1c[!a1c$numcycles==1,],
                                   REML=FALSE)))

a1c$resRT[!a1c$numcycles==1] <- resid(RTmod0S2res)


sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="resRT", groupvars =c("totalConfpl"), na.rm=TRUE)

sum1$totalConfpl <- as.numeric(as.character(sum1$totalConfpl))
#levels(sum1$totalConfpl) <- levels(pSetConfS2$data$totalConfidence)
#sum1 <- sum1[!sum1$totalConfpl==-4,]
sum1$totalConfidence <- sum1$totalConfpl
sum1$fit <- sum1$resRT + 7.62414 # this is the intercept from the residual model


plmodfstbtTCS2wr <- plmodfstbtTCS2+ geom_point(data=sum1, aes(x=totalConfidence, y=fit)) +geom_line(data=sum1, aes(x=totalConfidence, y=fit), linetype = 2, size = 0.25) +geom_errorbar(data=sum1, aes(x=totalConfidence, max = fit + se, min = fit- se), width = 0.01)

# need ConfDif

a1c$ConfDifpl <- (round(a1c$ConfDif/2))*2

a1c$OVpl <- round(a1c$ASV/2.5)

sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="resC", groupvars =c("ConfDifpl", "OVpl" ), na.rm=TRUE)
sum1$OVpl <- as.numeric(as.character(sum1$OVpl))*2.5
sum1$OVpl <- (sum1$OVpl-mean(a1c$ASV))/10
sum1$ConfDifpl <- as.factor(sum1$ConfDifpl)
levels(sum1$ConfDifpl) <-  levels(pConfDifS2$data$ConfDif)
sum1$ConfDif <- sum1$ConfDifpl
sum1 <- sum1[!sum1$N<10,]
sum1$resC <- sum1$resC+0.5
sum1$fit <-sum1$resC


pConfDifS2wr <- pConfDifS2+ geom_point(data=sum1, aes(x=OVpl, y=fit, color= ConfDif )) +geom_line(data=sum1, aes(x=OVpl, y=fit, color= ConfDif ), linetype = 2, size = 0.25) +geom_errorbar(data=sum1, aes(x=OVpl, max = fit + se, min = fit- se, color = ConfDif), width = 0.01)

### Fig 5

#plmodfstbt,plmodfstbtCB

# choice
a1c$ConfBiaspl  <- cut(a1c$ConfBias,
                       quantile(a1c$ConfBias, seq(0, 1, 1/5), na.rm=TRUE), include.lowest=T)#,

a1c$RVpl2 <- ((round((a1c$RVpl*10)/5))*5)/10
sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="isfirstIchosen", groupvars =c("ConfBiaspl", "RVpl2" ), na.rm=TRUE)
sum1$RVpl2 <- as.numeric(as.character(sum1$RVpl2))
#sum1$RVpl <- (sum1$OVpl-mean(a1c$ASV))/10
sum1$fstosnd <- sum1$RVpl2
sum1$ConfBiaspl <- as.factor(sum1$ConfBiaspl)
levels(sum1$ConfBiaspl) <- levels(plmodfstbt$data$cConfBias)
#sum1 <- sum1[!sum1$totalConfpl==-4,]
sum1$totalConfidence <- sum1$ConfBiaspl
sum1$fit <-sum1$isfirstIchosen

plmodfstbtwr <- plmodfstbt+ geom_point(data=sum1, aes(x=fstosnd, y=fit, color= ConfBiaspl )) +geom_line(data=sum1, aes(x=fstosnd, y=fit, color= ConfBiaspl ), linetype = 2, size = 0.25) +geom_errorbar(data=sum1, aes(x=fstosnd, max = fit + se, min = fit- se, color = ConfBiaspl), width = 0.01)

# RT

a1c$lRT <- log(a1c$RT*1000)
sum1 <-summarySE(a1c[! a1c$numcycles==1,], measurevar="lRT", groupvars =c("ConfBiaspl"), na.rm=TRUE)
levels(sum1$ConfBiaspl) <- levels(plmodfstbt$data$cConfBias)
sum1$ConfBiaspl <- as.numeric(as.character(sum1$ConfBiaspl))
#levels(sum1$totalConfpl) <- levels(pSetConfS2$data$totalConfidence)
#sum1 <- sum1[!sum1$totalConfpl==-4,]
sum1$cConfBias <- sum1$ConfBiaspl
sum1$fit <- sum1$lRT


plmodfstbtCBwr <- plmodfstbtCB+ geom_point(data=sum1, aes(x=cConfBias, y=fit)) +geom_line(data=sum1, aes(x=cConfBias, y=fit), linetype = 2, size = 0.25) +geom_errorbar(data=sum1, aes(x=cConfBias, max = fit + se, min = fit- se), width = 0.01)




######################### simulations & Combi plots ######################

# Study 1


print (summary(Choicemod0SIMb <- glm(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV, data = aSIMb, 
                                     family=binomial(link='logit') ))) 

tab_model(Choicemod0SIMb, transform=NULL)


## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0SIMb )

IA <- as.data.frame(eff_df)
pSPDSIM1<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))



## ... but the the relationship between presentation duration and choice depends on value:
# participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
eff_df <- Effect(c("savV", "spdfirst"), Choicemod0SIMb, xlevels=list(savV =seq(min(aSIMb$savV ), max(aSIMb$savV), 0.1), spdfirst =seq(min(aSIMb$spdfirst, na.rm=TRUE ), max(aSIMb$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVSIM1 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="none") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 


## RT

print (summary(RTmod0SIM1 <- lm(log(RT*1000)~ sVD + savV , aSIMb, 
                                REML=FALSE)))

tab_model(RTmod0SIM1)


########################### flat prior model  and zero prior model ################################

print (summary(Choicemod0SIMbfp <- glm(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV , data = aSIMbfp, 
                                       family=binomial(link='logit') ))) 

print (summary(Choicemod0SIMbzp <- glm(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV, data = aSIMbzp, 
                                       family=binomial(link='logit') ))) 
tab_model(Choicemod0SIMbfp, Choicemod0SIMbzp, transform=NULL)


## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0SIMbfp)

IA <- as.data.frame(eff_df)
pSPDSIM1fp<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

eff_df <- Effect(c("spdfirst"), Choicemod0SIMbzp,xlevels=list(spdfirst=seq(min(aSIMbzp$spdfirst), max(aSIMbzp$spdfirst), 0.1))  )

IA <- as.data.frame(eff_df)
pSPDSIM1zp<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))



## ... but the the relationship between presentation duration and choice depends on value:
# participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
eff_df <- Effect(c("savV", "spdfirst"), Choicemod0SIMbfp, xlevels=list(savV =seq(min(aSIMbfp$savV ), max(aSIMbfp$savV), 0.1), spdfirst =seq(min(aSIMbfp$spdfirst, na.rm=TRUE ), max(aSIMbfp$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVSIM1fp <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.8, 0.25)) +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0SIMbzp, xlevels=list(savV =seq(min(aSIMbzp$savV ), max(aSIMbzp$savV), 0.1), spdfirst =seq(min(aSIMbzp$spdfirst, na.rm=TRUE ), max(aSIMbzp$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVSIM1zp <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="none") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 

################### Plot Figure 3 ##########################
pdf(paste0(basepath, "/Figures/Combi/Fig3old.pdf"), width = 20, height = 8)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstOVSIM1, pSPDSIM1,plmodfstOVS1wr,pSPDS1wd,plmodfstOVS2wr, pSPDS2wd, plmodfstOVSIM1zp, pSPDSIM1zp,plmodfstOVSIM1fp,  pSPDSIM1fp, cols=5)
dev.off()


pdf(paste0(basepath, "/Figures/Combi/Fig3.pdf"), width = 20, height = 8)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pfirstlongSIMb, pSPDSIM1,pfirstlongS1,pSPDS1wd,pfirstlongS2, pSPDS2wd, pfirstlongSIMbzp, pSPDSIM1zp,pfirstlongSIMbfp,  pSPDSIM1fp, cols=5)
dev.off()

## RT
print (summary(RTmod0SIM1fp <- lm(log(RT*1000)~ sVD + savV , aSIMbfp, 
                                  REML=FALSE)))
print (summary(RTmod0SIM1zp <- lm(log(RT*1000)~ sVD + savV , aSIMbzp, 
                                  REML=FALSE)))
tab_model(RTmod0SIM1fp, RTmod0SIM1zp)


# Study 2

# main
print (summary(Choicemod0SIM2 <- glm(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV , data = aSIMc, #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                     family=binomial(link='logit')))) 

tab_model(Choicemod0SIM2, transform =NULL)



eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0SIM2, xlevels=list(fstosnd =seq(min(aSIMc$fstosnd ), max(aSIMc$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)
pSetConfSIM2<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0SIM2, xlevels=list(savV =seq(min(aSIMc$savV ), max(aSIMc$savV), 0.1)) )


IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)
pConfDifSIM2<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


## how about RT?


print (summary(RTmod0SIM2 <- lm(log(RT*1000)~   sVD + savV + ConfDif + totalConfidence, aSIMc,
                                REML=FALSE)))

tab_model(RTmod0SIM2)

eff_df <- Effect(c("totalConfidence"), RTmod0SIM2 )

IA <- as.data.frame(eff_df)
plmodfstbtTCSIM2 <- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+#ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 


#################### AC #######################


print (summary(Choicemod0SIM2ac <- glm(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV , data = aSIMcac, #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                       family=binomial(link='logit')))) 

tab_model(Choicemod0SIM2, Choicemod0SIM2ac, transform =NULL)



eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0SIM2ac, xlevels=list(fstosnd =seq(min(aSIMcac$fstosnd ), max(aSIMcac$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)
pSetConfSIM2ac<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0SIM2ac, xlevels=list(savV =seq(min(aSIMcac$savV ), max(aSIMcac$savV), 0.1)) )


IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)
pConfDifSIM2ac<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 



## how about RT?

print (summary(RTmod0SIM2ac <- lm(log(RT*1000)~   sVD + savV + ConfDif + totalConfidence, aSIMcac,
                                  REML=FALSE)))

tab_model(RTmod0SIM2ac)

eff_df <- Effect(c("totalConfidence"), RTmod0SIM2ac )

IA <- as.data.frame(eff_df)
plmodfstbtTCSIM2ac <- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

######## plot Figure 4 #############
# plot simulated data with human data
pdf(paste0(basepath, "/Figures/Combi/Fig4.pdf"), width = 12, height = 12)
multiplot(pSetConfSIM2, plmodfstbtTCSIM2,pConfDifSIM2,  pSetConfS2wr, plmodfstbtTCS2wr, pConfDifS2wr, pSetConfSIM2ac, plmodfstbtTCSIM2ac, pConfDifSIM2ac, cols=3)
dev.off()


####### Confidence bias

print (summary(Choicemod0wbSIM <- glm(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + cConfBias + savV + fstosnd:cConfBias + ConfDif:savV + spdfirst:savV 
                                      , data = aSIMcoc, 
                                      family=binomial(link='logit')))) 

#tab_model(Choicemod0wbSIM, transform =NULL)


print (summary(RTmod0wbSIM <- lm(log(RT*1000)~  sVD + savV + ConfDif + totalConfidence +  +cConfBias  , data = aSIMcoc, 
                                 REML=FALSE)))


eff_df <- Effect(c("cConfBias"), RTmod0wbSIM )

IA <- as.data.frame(eff_df)
plmodfstbtCBSIM<- ggplot(data=IA, aes(x=cConfBias, y=fit)) + geom_line()+theme_bw(12)+ geom_ribbon(data=IA, aes(x=cConfBias, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(6.5, 7.9)+
  xlab("Confidence Bias") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("fstosnd", "cConfBias"), Choicemod0wbSIM, xlevels=list(fstosnd =seq(min(aSIMcoc$fstosnd ), max(aSIMcoc$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cConfBias <- as.factor(IA$cConfBias)
plmodfstbtSIM<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= cConfBias)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=greyssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = cConfBias),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Bias", values=greyssub)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1))



pdf(paste0(basepath, "/Figures/Combi/Fig5.pdf"), width = 8, height = 8)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstbtSIM,plmodfstbtCBSIM, plmodfstbtwr,plmodfstbtCBwr,  cols=2)
dev.off()

############# compare Study 1 and Study 2 ################


a1d <- a1[!a1$Choice==-1,]

a1d$spdfirst<- a1d$pdfirst - 0.5
a1d$cRT <- scale(a1d$RT, scale=FALSE, center=TRUE)

a1d$Version <- as.factor(a1d$Version)
contrasts(a1d$Version) <- contr.sdif(2)
a1b$savV <- scale(a1b$ASV/10, scale=FALSE, center=TRUE)

a1b$sfstItemV <- scale(a1b$fstItemV, scale=FALSE, center=TRUE)
a1b$ssndItemVal <- scale(a1b$sndItemVal, scale=FALSE, center=TRUE)

# 1) How do values and presentation duration affect choice?

# we don't let the model fit a random intercept because that won't converge. There's no variance in bias.
# print (summary(Choicemod0 <- buildmer(isfirstIchosen ~ (fstosnd + savV)* spdfirst+cRT  +(0+(fstosnd + savV)* spdfirst+cRT |SubNum), data = a1b, 
#                                    family = binomial))) 

## this is the  model we get from buildmer. That doesn't allow us to test the interaction of presentation duration with relative value, hence we run the one below 
print (summary(Choicemod0ALL <- glmer(isfirstIchosen ~ (fstosnd + spdfirst + cRT + savV + spdfirst:savV)*Version +      (1 + fstosnd + spdfirst | SubNum), data = a1d[!a1d$numcycles==1,], 
                                      family = binomial))) 

tab_model(Choicemod0ALL, transform=NULL)

print (summary(Choicemod0ALLn <- glmer(isfirstIchosen ~ Version/(fstosnd + spdfirst + cRT + savV + spdfirst:savV) +      (1 + fstosnd + spdfirst | SubNum), data = a1d[!a1d$numcycles==1,], 
                                       family = binomial))) 

tab_model(Choicemod0ALLn, transform=NULL)

print (summary(RTmod0ALL <- lmer(log(RT*1000)~  (sVD + savV)*Version  +  (1 + sVD + savV | SubNum), a1d[!a1d$Choice==-1,], 
                                 REML=FALSE)))

tab_model(RTmod0ALL)


## RT distributions:

a1bt <- a1b
a1bt$RT[a1bt$isFirstChosen ==0] <- a1bt$RT[a1bt$isFirstChosen ==0]*-1
ggplot(data= a1bt, aes(x = RT, group = isfirstIchosen, fill=isfirstIchosen)) +geom_density()+ theme_classic() 

