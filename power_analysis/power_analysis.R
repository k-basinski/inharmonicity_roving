library(lme4)

pwr_calc <- function(b0,b1,b0_sd,res_sd,nsims,ssizes) {
  # intialize data frame to store the output 
  pwr <- data.frame() 
  
  for (s in 1:length(ssizes)) {
    for (n in 1:nsims) {
      # df index
      idx <- (s-1) * nsims + n
      
      # current sample size
      ssize <- ssizes[s]
      
      # conditions factor
      conds <- rep(c(0,1,2),ssize)
      
      # subject id's
      subs <- rep(1:ssize, each = length(unique(conds)))
      
      # add intercept
      intercept <- rep(rnorm(ssize,b0,b0_sd), each = length(unique(conds)))
      
      # add condition effect
      beta1 <- rep(b1,each= length(unique(conds)))*conds
      
      # add residual noise
      residuals <- rnorm(length(subs),0,res_sd)
      
      # collect in a dataframe and calculate simulated measured outcome (y)
      d <- data.frame('cond' = as.character(conds), 
                      'sub' = subs,
                      'b0' = intercept,
                      'b1' = beta1,
                      'res' = residuals,
                      'y' = intercept + beta1 + residuals)
      
      # fit models
      m0 <- lmer(y~1 + (1|sub), data = d, REML = FALSE)
      m1 <- lmer(y~cond + (1|sub), data = d, REML = FALSE)
      
      # perform likelihood ratio test
      test <- anova(m0,m1)
      
      #store output of simulation
      pwr[idx,'sim'] <- n
      pwr[idx, 'ssize'] <- ssize
      pwr[idx, 'b0'] <- summary(m1)$coefficients[1]
      pwr[idx, 'b1'] <- summary(m1)$coefficients[2]
      pwr[idx, 'sd_int'] <- attr(summary(m1)$varcor$sub,"stddev") 
      pwr[idx, 'sd_res'] <- summary(m1)$sigma
      pwr[idx, 'x2'] <- test$Chisq[2]
      pwr[idx, 'p'] <- test$`Pr(>Chisq)`[2]
    }
  }
  return(pwr)
}
 

e0 <- 3 # uV # uV # intercept (in fT or micro Volts)
e1 <- 1 # -1 uv # uV # minimum difference between conditions
e0_sd <- 0.95 # 0.95 uV # standard deviation of the intercept 
eres_sd <- 1.5 # 1.5 uV # residual standard deviation
nsims <- 500 # number of simulations per sample size
ssizes <- 10:40 # sample sizes

# set.seed(900509)
pwr2 <- pwr_calc(e0,e1,e0_sd,eres_sd,nsims,ssizes)

pwr2

summary2 <- aggregate(pwr2$p,by = list(pwr2$ssize), FUN = function(x) sum(x < 0.05)/length(x))
colnames(summary2) <- c('sample.size','power')
print(summary2)


with(summary2, plot(sample.size, power, type = 'ol')) 
title('Power curve - EEG data')
