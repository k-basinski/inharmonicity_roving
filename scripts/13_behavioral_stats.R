library(readr)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(broom)
library(ordinal)

# load data
df_t1 <- read_csv("paradigms/behavioral/data/behavioral_task1.csv")
df_t2 <- read_csv("paradigms/behavioral/data/behavioral_task2.csv")


# set factors
cond_levels <- c("harmonic", "inharmonic", "changing")
response_levels <- c("One sound", "Two sounds", "Three or more sounds")
df_t1$condition <- factor(df_t1$condition, levels=cond_levels)
df_t1$response <- factor(df_t1$response, levels=response_levels)

# make contingency table and chisquare the results
cont_table <- table(df_t1$condition, df_t1$response)
cont_table %>% round(2) %>% print()
prop_table <- prop.table(cont_table)
prop_table %>% round(2) %>% print()

# make cumulative-link mixed models for Task 1
f0 <- "response ~ (1|pid)"
f1 <- "response ~ condition + (1|pid)"
m0 <- clmm(f0, data=df_t1)
m1 <- clmm(f1, data=df_t1)
summary(m1)


# model task 2 results
f0 <- "abs_error ~ (1|pid)"
f1 <- "abs_error ~ condition + (1|pid)"
m0 <- lmer(f0, data=df_t2)
m1 <- lmer(f1, data=df_t2)

AIC(m0, m1)
print(anova(m0, m1, refit = FALSE))
anova(m1)
emm <- emmeans(m1, "condition")
pairs(emm) %>%
    broom::tidy() %>%
    print()

emm
