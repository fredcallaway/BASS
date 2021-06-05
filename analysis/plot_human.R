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
# fileLoc <- dirname(rstudioapi::getSourceEditorContext()$path)
# setwd(fileLoc) # go to script location first
basepath <- paste0(getwd(), '/../')

didLmerConverge = function(lmerModel){
  relativeMaxGradient=signif(max(abs(with(lmerModel@optinfo$derivs,solve(Hessian,gradient)))),3)
  if (relativeMaxGradient < 0.001) {
    cat(sprintf("\tThe relative maximum gradient of %s is less than our 0.001 criterion.\n\tYou can safely ignore any warnings about a claimed convergence failure.\n\n", relativeMaxGradient))
  }
  else {
    cat(sprintf("The relative maximum gradient of %s exceeds our 0.001 criterion.\nThis looks like a real convergence failure; maybe try simplifying your model?\n\n", relativeMaxGradient))
  }
}

### load data ###
# choice data
input_file = paste0(basepath, 'data/allSubDataTablePilot.xls')
input_file
a1 = read.xls(input_file)
# Rating data
input_file =paste0(basepath, 'data/allSubRateDataTablePilot.xls')
aR = read.xls(input_file)
aR <- aR[!is.na(aR$RateRT1),]


### color definitions ###
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
## palette for relative presentation effects
darkcols <- brewer.pal(8, "RdGy")
darkcolssub <-  darkcols[c(1:2, 6:8)]
# Confidence palette
darkcolsg <- brewer.pal(8, "Greens")
darkcolssubg <-  darkcolsg[c(4:8)]

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

##### main choice analyses ####

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
print (summary(Choicemod0best <- glmer(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV +      (0 + fstosnd + spdfirst | SubNum), data = a1b, 
                                   family = binomial))) 

sv1_max <- svd(getME(Choicemod0best, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

# tab_model(Choicemod0best, transform=NULL)


print (summary(Choicemod0 <- glmer(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV + spdfirst:fstosnd+     (0 + fstosnd + spdfirst | SubNum), data = a1b, 
                                   family = binomial))) 

sv1_max <- svd(getME(Choicemod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1) 
# tab_model(Choicemod0, transform=NULL)

## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0, xlevels=list(fstosnd =seq(min(a1b$fstosnd, na.rm=TRUE ), max(a1b$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pRVS1<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(basepath, "figures/RV503.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pRVS1
dev.off()

## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0 )

IA <- as.data.frame(eff_df)
pSPDS1<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(basepath, "figures/RPDBias503.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pSPDS1
dev.off()


## ... but the the relationship between presentation duration and choice depends on value:
# participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
eff_df <- Effect(c("savV", "spdfirst"), Choicemod0, xlevels=list(savV =seq(min(a1b$savV ), max(a1b$savV), 0.1), spdfirst =seq(min(a1b$spdfirst, na.rm=TRUE ), max(a1b$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVS1 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 
pdf(paste0(basepath, "figures/RPDbyOV503.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
plmodfstOVS1
dev.off()


## overall people are more likely to choose the first item when they make fast decisions
eff_df <- Effect(c("cRT"), Choicemod0, xlevels=list(cRT =seq(min(a1b$cRT, na.rm=TRUE ), max(a1b$cRT, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cRT <- IA$cRT - min(a1b$cRT, na.rm=TRUE )
pRTS1 <- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black") + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(basepath, "figures/RTonBias503.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pRTS1
dev.off()

## so are presentation duration and RT correlated? No. But they do on average see the first item more than the second one and that could explain the bias
print (summary(PDMOD <- lmer(spdfirst~ cRT +(1 |SubNum), a1b[!a1b$Choice==-1,], 
                             REML=FALSE)))

sv1_max <- svd(getME(PDMOD, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

# yep. 
print (summary(ChoiceJustRT <- glmer(isfirstIchosen ~ cRT  +(0+cRT|SubNum), data = a1b, 
                                     family = binomial))) 

## How about response times?
# is there a better way to code this? What we really want to know is 

# print (summary(RTmod0 <- buildmer(log(RT*1000)~ (sVD+savV+fstosnd)*spdfirst+((sVD+savV+fstosnd)*spdfirst|SubNum), a1b[!a1b$Choice==-1,], 
#                               REML=FALSE)))


print (summary(RTmod0 <- lmer(log(RT*1000)~ spdfirst + sVD + savV + fstosnd + spdfirst:fstosnd +  
                                (1 + sVD + savV + spdfirst | SubNum), a1b[!a1b$Choice==-1,], 
                              REML=FALSE)))

sv1_max <- svd(getME(RTmod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

didLmerConverge(RTmod0)

# tab_model(RTmod0)

eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0, xlevels=list(savV =seq(min(a1b$fstosnd ), max(a1b$fstosnd), 0.1), spdfirst =seq(min(a1b$spdfirst, na.rm=TRUE ), max(a1b$spdfirst, na.rm=TRUE), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodRTRVPDS1 <- ggplot(data=IA, aes(x=fstosnd, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + geom_vline(xintercept=0, linetype=2, size=0.2)+ xlab("Relative value") + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") 
pdf(paste0(basepath, "figures/RPDbyRVRT503.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
plmodRTRVPDS1
dev.off()



### Now we do that same thing for Version 504

a1c <- a1c[!a1c$Choice==-1,]

# 1) How do values and presentation duration affect choice?

## original
# print (summary(Choicemod0 <- glmer(isfirstIchosen ~ (fstosnd + savV)* spdfirst +cRT  +(0+fstosnd+spdfirst|SubNum), data = a1c[!a1c$numcycles==1,], 
#                                    family = binomial))) 
# 
# sv1_max <- svd(getME(Choicemod0, "Tlist")[[1]]) 
# sv1_max$d
# round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  
# 
# tab_model(Choicemod0)
# 
# library(jtools)
# summ(Choicemod0, scale=TRUE)
# 
# ## people's choices are consistent with the ratings they gave us earlier
# eff_df <- Effect(c("fstosnd"), Choicemod0, xlevels=list(fstosnd =seq(min(a1c$fstosnd, na.rm=TRUE ), max(a1c$fstosnd, na.rm=TRUE), 0.1)) )
# 
# IA <- as.data.frame(eff_df)
# pRV<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
#   scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
#   coord_cartesian(ylim = c(0, 1)) 
# 
# pdf(paste0(basepath, "figures/RV504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# pRV
# dev.off()
# 
# ## There is no main effect of presentation duration on choice
# eff_df <- Effect(c("spdfirst"), Choicemod0 )
# 
# IA <- as.data.frame(eff_df)
# pRT<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
#   scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))
# 
# pdf(paste0(basepath, "figures/RPDBias504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# pRT
# dev.off()
# 
# 
# ## ... but the the relationship between presentation duration and choice depends on value:
# # participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
# eff_df <- Effect(c("savV", "spdfirst"), Choicemod0, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )
# IA <- as.data.frame(eff_df)
# IA$spdfirst <- as.factor(round(IA$spdfirst,1))
# 
# plmodfstOV <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
#   scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
#   coord_cartesian(ylim = c(0, 1)) 
# pdf(paste0(basepath, "figures/RPDbyOV504.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# plmodfstOV
# dev.off()
# 
# 
# ## overall people are more likely to choose the first item when they make fast decisions
# eff_df <- Effect(c("cRT"), Choicemod0, xlevels=list(cRT =seq(min(a1c$cRT, na.rm=TRUE ), max(a1c$cRT, na.rm=TRUE), 0.1)) )
# 
# IA <- as.data.frame(eff_df)
# IA$cRT <- IA$cRT - min(a1c$cRT, na.rm=TRUE )
# pRT<- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
#   scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black") + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))
# 
# pdf(paste0(basepath, "figures/RTonBias504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# pRT
# dev.off()

## so are presentation duration and RT correlated? No. But they do on average see the first item more than the second one and that could explain the bias
print (summary(PDMOD <- lmer(spdfirst~ cRT +(1 |SubNum), a1c[!a1c$numcycles==1,], 
                             REML=FALSE)))

sv1_max <- svd(getME(PDMOD, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

# yep. 
print (summary(ChoiceJustRT <- glmer(isfirstIchosen ~ cRT  +(0+cRT|SubNum), data = a1c[!a1c$numcycles==1,], 
                                     family = binomial))) 

## How about response times?
# print (summary(RTmod0 <- lmer(log(RT*1000)~ sVD+savV+(fstosnd)*spdfirst+(sVD|SubNum), a1c[!a1c$numcycles==1,], 
#                               REML=FALSE)))
# 
# sv1_max <- svd(getME(RTmod0, "Tlist")[[1]]) 
# sv1_max$d
# round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  
# 
# didLmerConverge(RTmod0)
# 
# eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0, xlevels=list(savV =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1), spdfirst =seq(min(a1c$spdfirst ), max(a1c$spdfirst), 0.2)) )
# IA <- as.data.frame(eff_df)
# IA$spdfirst <- as.factor(round(IA$spdfirst,1))
# 
# plmodfstOV <- ggplot(data=IA, aes(x=fstosnd, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
#   scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + geom_vline(xintercept=0, linetype=2, size=0.2)+ xlab("Relative value") + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") 
# pdf(paste0(basepath, "figures/RPDbyRVRT504.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
# plmodfstOV
# dev.off()

## ok, cool. So, how about confidence?

# How does Confidence affect choices?


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
# tab_model(Choicemod0S2, transform =NULL)



## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0S2, xlevels=list(fstosnd =seq(min(a1c$fstosnd, na.rm=TRUE ), max(a1c$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pRVS2<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(basepath, "figures/RV504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pRVS2
dev.off()

## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0S2 )

IA <- as.data.frame(eff_df)
pPDS2<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(basepath, "figures/RPDBias504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pPDS2
dev.off()


## overall people are more likely to choose the first item when they make fast decisions
eff_df <- Effect(c("cRT"), Choicemod0S2, xlevels=list(cRT =seq(min(a1c$cRT, na.rm=TRUE ), max(a1c$cRT, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cRT <- IA$cRT - min(a1c$cRT, na.rm=TRUE )
pRTS2<- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black") + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(basepath, "figures/RTonBias504.pdf"), width = 4, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
pRTS2
dev.off()

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0S2, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1), spdfirst =seq(min(a1c$spdfirst, na.rm = TRUE ), max(a1c$spdfirst, na.rm = TRUE), 0.2)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))
plmodfstOVS2 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 
pdf(paste0(basepath, "figures/RPDbyOV504ControllingForConf.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
plmodfstOVS2
dev.off()



eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0S2, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)
pSetConfS2<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0S2, xlevels=list(savV =seq(min(a1c$savV ), max(a1c$savV), 0.1)) )


IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)
pConfDifS2<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssubg)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 

#plmodfstbt,
pdf(paste0(basepath, "figures/OVByConf.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot( pSetConfS2, pConfDifS2, cols=3)
dev.off()


## how about RT?
#a1c$logRT <- log(a1c$RT*1000)


# 
# print (summary(RTmod0 <- buildmer(log(RT*1000)~  sVD+savV+fstosnd*(spdfirst)+(totalConfidence+ConfDif)+(sVD+savV+fstosnd*(spdfirst)+(totalConfidence+ConfDif)|SubNum), a1c[!a1c$numcycles==1,], 
#                               REML=FALSE)))



 print (summary(RTmod0S2 <- lmer(log(RT*1000)~   sVD + savV + ConfDif + totalConfidence + fstosnd + spdfirst + fstosnd:spdfirst+(savV+sVD|SubNum), a1c[!a1c$numcycles==1,],
                               REML=FALSE)))

 sv1_max <- svd(getME(RTmod0S2, "Tlist")[[1]])
 sv1_max$d
 round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)

# tab_model(RTmod0S2)


eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0S2, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(IA$spdfirst)

plmodfstS2 <- ggplot(data=IA, aes(x=fstosnd, y=fit, color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se,  fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+#ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) +theme(strip.background = element_rect(colour="white",fill="white"))+ theme(legend.position="right") #c(0.8, 0.2)

pdf(paste0(basepath, "figures/RPDbyRVRT504.pdf"), width = 5, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
plmodfstS2
dev.off()

eff_df <- Effect(c("ConfDif"), RTmod0S2 )

IA <- as.data.frame(eff_df)
plmodfstbtCDS2 <- ggplot(data=IA, aes(x=ConfDif, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=ConfDif, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Difference") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("totalConfidence"), RTmod0S2 )

IA <- as.data.frame(eff_df)
plmodfstbtTCS2 <- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(basepath, "figures/RTConfAll.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstbtTCS2, plmodfstbtCDS2,  cols =2)
dev.off()


##################### Bias analyses #######################
#fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum)

print (summary(Choicemod0wb <- glmer(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + cConfBias + savV + fstosnd:cConfBias + ConfDif:savV + spdfirst:savV +      (1 + fstosnd | SubNum)
, data = a1c[!a1c$numcycles==1,], #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                     family = binomial))) 

sv1_max <- svd(getME(Choicemod0, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

# tab_model(Choicemod0wb, transform =NULL)


# #sVD + savV + ConfDif + totalConfidence + fstosnd + spdfirst + fstosnd:spdfirst+(savV+sVD|SubNum) 

print (summary(RTmod0wb <- lmer(log(RT*1000)~  sVD + savV + ConfDif + totalConfidence + fstosnd + spdfirst +cConfBias + fstosnd:spdfirst+(savV+sVD|SubNum) , a1c[!a1c$numcycles==1,], 
                                REML=FALSE)))

sv1_max <- svd(getME(RTmod0wb, "Tlist")[[1]]) 
sv1_max$d
round(sv1_max$d^2/sum(sv1_max$d^2)*100, 1)  

# tab_model(RTmod0wb)



eff_df <- Effect(c("cConfBias"), RTmod0wb )

IA <- as.data.frame(eff_df)
plmodfstbtCB<- ggplot(data=IA, aes(x=cConfBias, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cConfBias, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Bias") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("fstosnd", "cConfBias"), Choicemod0wb, xlevels=list(fstosnd =seq(min(a1c$fstosnd ), max(a1c$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cConfBias <- as.factor(IA$cConfBias)
plmodfstbt<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= cConfBias)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = cConfBias),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1))

multiplot(plmodfstbtCB, plmodfstbt, cols=2)

save.image(file="plot_human.Rdata")
