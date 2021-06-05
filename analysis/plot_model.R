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
library(RColorBrewer)
library(sjPlot)
library(buildmer)
library(effects)
library(glue)
# library(ggplot2)
load("plot_human.RData")

# %% ==================== Load and preprocess data ====================

## load simulation data
# input_file =paste0(basepath, 'model/results/qualitative_sim_may6.csv')
basepath <- paste0(getwd(), '/../')
version = "v3"
input_file = glue('{basepath}model/results/qualitative_sim_{version}.csv')
aSIMb <-   read.csv(input_file)


# Study 2
input_file = glue('{basepath}model/results/qualitative_sim_{version}.csv')
aSIMc <-   read.csv(input_file)

figpath = glue('{basepath}figures/Combi_{version}/')
figpath
dir.create(paste0(figpath, "503"), recursive=T)
dir.create(paste0(figpath, "504"), recursive=T)

#length(aSIM$subject[aSIM$subject==1064])

### compute same vars for simulated data
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


### compute same vars for simulated data
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
{
  aSIMc$ConfBias[aSIMc$SubNum==i]  <-  mean(c(aSIMc$conf1[aSIMc$SubNum==i], aSIMc$conf2[aSIMc$SubNum==i]))
}

aSIMc$cConfBias <- scale(aSIMc$ConfBias, scale=FALSE, center= TRUE)
aSIMc$sumConfidence <- aSIMc$conf1+aSIMc$conf2
aSIMc$totalConfidence <- (aSIMc$sumConfidence)/2 - aSIMc$ConfBias
# aSIMc$totalConfidence <- (aSIMc$sumConfidence)/2


aSIMc$cfstConfidence <- scale(aSIMc$conf1, scale=FALSE, center = TRUE)
aSIMc$csndConfidence <- scale(aSIMc$conf2, scale=FALSE, center = TRUE)
aSIMc$ConfDif <- aSIMc$conf1-aSIMc$conf2# don't center

mean(a1c$totalConfidence, na.rm=T)
mean(aSIMc$totalConfidence, na.rm=T)

head(a1c$totalConfidence)


# %% ==================== Make plots ====================

# Study 1


Choicemod0SIMb <- glm(isfirstIchosen ~ fstosnd + spdfirst + cRT + savV + spdfirst:savV + spdfirst:fstosnd, data = aSIMb, 
                                 family=binomial(link='logit') )

# tab_model(Choicemod0SIMb, transform=NULL)

## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0SIMb, xlevels=list(fstosnd =seq(min(aSIMb$fstosnd, na.rm=TRUE ), max(aSIMb$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pRVSIM1<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(figpath, "503/RV503SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pRVS1, pRVSIM1, cols = 2 )
dev.off()

## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0SIMb )

IA <- as.data.frame(eff_df)
pSPDSIM1<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(figpath, "503/RPDBias503SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pSPDS1, pSPDSIM1, cols =2)
dev.off()


## ... but the the relationship between presentation duration and choice depends on value:
# participants are more likely to choose the item they saw more when option values are high, but less likely when option values are low
eff_df <- Effect(c("savV", "spdfirst"), Choicemod0SIMb, xlevels=list(savV =seq(min(aSIMb$savV ), max(aSIMb$savV), 0.1), spdfirst =seq(min(aSIMb$spdfirst, na.rm=TRUE ), max(aSIMb$spdfirst, na.rm=TRUE ), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodfstOVSIM1 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 
pdf(paste0(figpath, "503/RPDbyOV503SIM.pdf"), width = 10, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstOVS1,plmodfstOVSIM1, cols=2)
dev.off()


## overall people are more likely to choose the first item when they make fast decisions
eff_df <- Effect(c("cRT"), Choicemod0SIMb, xlevels=list(cRT =seq(min(a1b$cRT, na.rm=TRUE ), max(a1b$cRT, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cRT <- IA$cRT - min(a1b$cRT, na.rm=TRUE )
pRTSIM1 <- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black") + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(figpath, "503/RTonBias503SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pRTS1,pRTSIM1, cols=2)
dev.off()

## RT

RTmod0SIM1 <- lm(log(RT*1000)~ spdfirst + sVD + savV + fstosnd + spdfirst:fstosnd, aSIMb, 
                              REML=FALSE)

# tab_model(RTmod0SIM1)

eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0SIM1, xlevels=list(savV =seq(min(aSIMb$fstosnd ), max(aSIMb$fstosnd), 0.1), spdfirst =seq(min(aSIMb$spdfirst, na.rm=TRUE ), max(aSIMb$spdfirst, na.rm=TRUE), 0.2)) )
IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))

plmodRTRVPDSIM1 <- ggplot(data=IA, aes(x=fstosnd, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + geom_vline(xintercept=0, linetype=2, size=0.2)+ xlab("Relative value") + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") 
pdf(paste0(figpath, "503/RPDbyRVRT503SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodRTRVPDS1, plmodRTRVPDSIM1, cols=2)
dev.off()

# Study 2


Choicemod0SIM2 <- glm(isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence +  fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV , data = aSIMc, #  & a1c$fstConfidence >0 & a1c$sndConfidence >0
                                   family=binomial(link='logit'))

# tab_model(Choicemod0SIM2, transform =NULL)



## people's choices are consistent with the ratings they gave us earlier
eff_df <- Effect(c("fstosnd"), Choicemod0SIM2, xlevels=list(fstosnd =seq(min(aSIMc$fstosnd, na.rm=TRUE ), max(aSIMc$fstosnd, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
pRVSIM2<- ggplot(data=IA, aes(x=fstosnd, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Relative first item value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))+
  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(figpath, "504/RV504SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pRVS2,pRVSIM2, cols=2)
dev.off()

## There is no main effect of presentation duration on choice
eff_df <- Effect(c("spdfirst"), Choicemod0SIM2 )

IA <- as.data.frame(eff_df)
pPDSIM2<- ggplot(data=IA, aes(x=spdfirst, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=spdfirst, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=spdfirst, y=0.5), size=0.2,linetype=2, color="black") + xlab("Relative Presentation Duration") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(figpath, "504/RPDBias504SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pPDS2, pPDSIM2, cols=2)
dev.off()


## overall people are more likely to choose the first item when they make fast decisions
eff_df <- Effect(c("cRT"), Choicemod0SIM2, xlevels=list(cRT =seq(min(aSIMc$cRT, na.rm=TRUE ), max(aSIMc$cRT, na.rm=TRUE), 0.1)) )

IA <- as.data.frame(eff_df)
IA$cRT <- IA$cRT - min(a1c$cRT, na.rm=TRUE )
pRTSIM2<- ggplot(data=IA, aes(x=cRT, y=fit  )) + geom_line()+scale_colour_manual(name="presentation\n duration\n first", values=cbPalette) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=cRT, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(0, 1)+
  scale_fill_manual(name="presentation\n duration\n first", values=cbPalette)+geom_line(data=IA, aes(x=cRT, y=0.5), size=0.2,linetype=2, color="black") + xlab("RT") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.25, 0.2))

pdf(paste0(figpath, "504/RTonBias504SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(pRTS2,pRTSIM2, cols=2)
dev.off()

eff_df <- Effect(c("savV", "spdfirst"), Choicemod0SIM2, xlevels=list(savV =seq(min(aSIMc$savV ), max(aSIMc$savV), 0.1), spdfirst =seq(min(aSIMc$spdfirst, na.rm = TRUE ), max(aSIMc$spdfirst, na.rm = TRUE), 0.2)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(round(IA$spdfirst,1))
plmodfstOVSIM2 <- ggplot(data=IA, aes(x=savV, y=fit , color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+# ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position="right") +#c(0.8, 0.2)
  coord_cartesian(ylim = c(0, 1)) 
pdf(paste0(figpath, "504/RPDbyOV504ControllingForConfSIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstOVS2, plmodfstOVSIM2, cols=2)
dev.off()


eff_df <- Effect(c("fstosnd", "totalConfidence"), Choicemod0SIM2, xlevels=list(fstosnd =seq(min(aSIMc$fstosnd ), max(aSIMc$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$totalConfidence <- as.factor(IA$totalConfidence)
pSetConfSIM2<- ggplot(data=IA, aes(x=fstosnd, y=fit , color= totalConfidence )) + geom_line()+scale_colour_manual(name="Set\n Confidence", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se, fill = totalConfidence),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Set\n Confidence", values=darkcolssubg)+geom_line(data=IA, aes(x=fstosnd, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("First minus Second Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 


eff_df <- Effect(c("savV", "ConfDif"), Choicemod0SIM2, xlevels=list(savV =seq(min(aSIMc$savV ), max(aSIMc$savV), 0.1)) )


IA <- as.data.frame(eff_df)
IA$ConfDif <- as.factor(IA$ConfDif)
pConfDifSIM2<- ggplot(data=IA, aes(x=savV, y=fit , color= ConfDif )) + geom_line()+scale_colour_manual(name="Confidence\n Difference", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=savV, max = fit + se, min = fit- se, fill = ConfDif),alpha=0.1, inherit.aes = FALSE)+ #ylim(0, 1)+
  scale_fill_manual(name="Confidence\n Difference", values=darkcolssubg)+geom_line(data=IA, aes(x=savV, y=0.5), size=0.2,linetype=2, color="black")+ geom_vline(xintercept=0, linetype=2, size=0.2) + xlab("Overall Value") + ylab("P(first chosen)") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.3, 0.25),legend.background=element_blank())+  coord_cartesian(ylim = c(0, 1)) 

#plmodfstbt,
pdf(paste0(figpath, "504/OVByConfSIM.pdf"), width = 8, height = 8)#, units = 'cm', res = 200, compression = 'lzw'
multiplot( pSetConfS2,pConfDifS2, pSetConfSIM2, pConfDifSIM2, cols=2)
dev.off()


## how about RT?

RTmod0SIM2 <- lm(log(RT*1000)~   sVD + savV + ConfDif + totalConfidence + fstosnd + spdfirst + fstosnd:spdfirst, aSIMc,
                                REML=FALSE)

# tab_model(RTmod0SIM2)


eff_df <- Effect(c("fstosnd", "spdfirst"), RTmod0SIM2, xlevels=list(fstosnd =seq(min(aSIMc$fstosnd ), max(aSIMc$fstosnd), 0.1)) )

IA <- as.data.frame(eff_df)
IA$spdfirst <- as.factor(IA$spdfirst)

plmodfstSIM2 <- ggplot(data=IA, aes(x=fstosnd, y=fit, color= spdfirst )) + geom_line()+scale_colour_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=fstosnd, max = fit + se, min = fit- se,  fill = spdfirst),alpha=0.1, inherit.aes = FALSE)+#ylim(-0.5, 1.5)+
  scale_fill_manual(name="Relative\nfirst item\npresentation", values=darkcolssub) + ylab("log RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) +theme(strip.background = element_rect(colour="white",fill="white"))+ theme(legend.position="right") #c(0.8, 0.2)

pdf(paste0(figpath, "504/RPDbyRVRT504SIM.pdf"), width = 8, height = 4)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstS2, plmodfstSIM2, cols=2)
dev.off()

eff_df <- Effect(c("ConfDif"), RTmod0SIM2 )

IA <- as.data.frame(eff_df)
plmodfstbtCDSIM2 <- ggplot(data=IA, aes(x=ConfDif, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=ConfDif, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Confidence Difference") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

eff_df <- Effect(c("totalConfidence"), RTmod0SIM2 )

IA <- as.data.frame(eff_df)
plmodfstbtTCSIM2 <- ggplot(data=IA, aes(x=totalConfidence, y=fit)) + geom_line()+scale_colour_manual(name="Confidence\n Bias", values=darkcolssubg) +theme_bw(12)+ geom_ribbon(data=IA, aes(x=totalConfidence, max = fit + se, min = fit- se),alpha=0.1, inherit.aes = FALSE)+ ylim(7.4, 7.75)+
  scale_fill_manual(name="Confidence\n Bias", values=darkcolssubg)+ xlab("Set Confidence") + ylab("RT") + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) + theme(legend.position=c(0.75, 0.25),legend.background=element_blank())#+  coord_cartesian(ylim = c(0, 1)) 

pdf(paste0(figpath, "504/RTConfAllSIM.pdf"), width = 8, height = 8)#, units = 'cm', res = 200, compression = 'lzw'
multiplot(plmodfstbtTCS2, plmodfstbtCDS2,plmodfstbtTCSIM2, plmodfstbtCDSIM2,  cols =2)
dev.off()



