library(readr)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(broom)

# change this to your repo location
setwd("/Users/kbas/cloud/sci/inharmonicity_roving/")

df <- read_csv("results/peak_measures.csv")
df

# filter only first deviants
dm1 <- df %>%
    filter(mismatch == "mismatch_1" & pitch_diff == "all")
dm1

dvs <- c(
    "mmn_mean_amp",
    "mmn_peak_lat",
    "p3_mean_amp",
    "p3_peak_lat"
)

for (dv in dvs) {
    paste("\n\n* ", dv, ":\n") %>%
        toupper() %>%
        cat()

    f0 <- paste(dv, " ~ (1 | pid)")
    f1 <- paste(dv, " ~ condition + (1 | pid)")
    m0 <- lmer(f0, data = dm1)
    m1 <- lmer(f1, data = dm1)
    AIC(m0, m1)
    print(anova(m0, m1, refit = TRUE))
    cat("\n")
    anova(m0, m1, refit = TRUE) %>%
        broom::tidy() %>%
        write_csv(paste("results/aov_", dv, ".csv", sep = ""))

    # estimated marginal means
    emm <- emmeans(m1, "condition")
    pairs(emm) %>%
        broom::tidy() %>%
        print()
    pairs(emm) %>%
        broom::tidy() %>%
        write_csv(paste("results/emmeans_", dv, ".csv", sep = ""))
    emm %>%
        broom::tidy() %>%
        print()
}

# see if p3 mean amplitude results is driven by one outlier

dm1 %>% filter(p3_mean_amp > 10)
# pid==3 is the outlier. Remove and remodel
dm1_no_outlier <- dm1 %>% filter(pid != 3)
dv <- "p3_mean_amp"
f0 <- paste(dv, " ~ (1 | pid)")
f1 <- paste(dv, " ~ condition + (1 | pid)")
m0 <- lmer(f0, data = dm1)
m1 <- lmer(f1, data = dm1)
AIC(m0, m1)
print(anova(m0, m1, refit = TRUE))
anova(m0, m1, refit = TRUE) %>%
    broom::tidy() %>%
    write_csv(paste("results/aov_outlier_p3.csv", sep = ""))
emm <- emmeans(m1, "condition")
pairs(emm)
pairs(emm) %>%
    broom::tidy() %>%
    write_csv(paste("results/emmeans_outlier_p3.csv", sep = ""))


# now the second deviants
dm2 <- df %>%
    filter(mismatch == "mismatch_2" & pitch_diff == "all")


for (dv in dvs) {
    paste("\n\n* ", dv, ":\n") %>%
        toupper() %>%
        cat()

    f0 <- paste(dv, " ~ (1 | pid)")
    f1 <- paste(dv, " ~ condition + (1 | pid)")
    m0 <- lmer(f0, data = dm2)
    m1 <- lmer(f1, data = dm2)
    AIC(m0, m1)
    print(anova(m0, m1, refit = TRUE))
    cat("\n")
    anova(m0, m1, refit = TRUE) %>%
        broom::tidy() %>%
        write_csv(paste("results/aov_", dv, "_m2.csv", sep = ""))

    # estimated marginal means
    emm <- emmeans(m1, "condition")
    pairs(emm) %>%
        broom::tidy() %>%
        print()
    pairs(emm) %>%
        broom::tidy() %>%
        write_csv(paste("results/emmeans_", dv, "_m2.csv", sep = ""))
}



# now the third deviants
dm3 <- df %>%
    filter(mismatch == "mismatch_3" & pitch_diff == "all")


for (dv in dvs) {
    paste("\n\n* ", dv, ":\n") %>%
        toupper() %>%
        cat()

    f0 <- paste(dv, " ~ (1 | pid)")
    f1 <- paste(dv, " ~ condition + (1 | pid)")
    m0 <- lmer(f0, data = dm3)
    m1 <- lmer(f1, data = dm3)
    AIC(m0, m1)
    print(anova(m0, m1, refit = TRUE))
    cat("\n")
    anova(m0, m1, refit = TRUE) %>%
        broom::tidy() %>%
        write_csv(paste("results/aov_", dv, "_m2.csv", sep = ""))

    # estimated marginal means
    emm <- emmeans(m1, "condition")
    pairs(emm) %>%
        broom::tidy() %>%
        print()
    pairs(emm) %>%
        broom::tidy() %>%
        write_csv(paste("results/emmeans_", dv, "_m2.csv", sep = ""))
}





# now with pitch diffs
df_pds <- df %>%
    filter(pitch_diff != "all" & mismatch == "mismatch_1") %>%
    mutate(pd = as.numeric(pitch_diff))

f0 <- "mmn_peak_lat ~ (1 | pid)"
f1 <- "mmn_peak_lat ~ pd + (1 | pid)"
f2 <- "mmn_peak_lat ~ pd + condition + (1 | pid)"
f3 <- "mmn_peak_lat ~ pd * condition + (1 | pid)"

fcontr <- "mmn_peak_lat ~ condition + (1 | pid)"


m0 <- lmer(f0, data = df_pds)
m1 <- lmer(f1, data = df_pds)
m2 <- lmer(f2, data = df_pds)
m3 <- lmer(f3, data = df_pds)
m_contr <- lmer(fcontr, data = df_pds)

summary(m0)
summary(m1)
summary(m2)
summary(m3)
summary(m_contr)

AIC(m0, m1, m2, m3, m_contr)

anova(m0, m1, m2, m3)
anova(m1, m2)
anova(m2, m3)

# including pitch difference does not improve model predictions
anova(m_contr, m2)
anova(m_contr, m3)

outputs = list()
for (dv in dvs) {
    print(dv)
    f1 <- paste(dv, " ~ condition + (1 | pid)")
    f2 <- paste(dv, " ~ condition + pd + (1 | pid)")
    f3 <- paste(dv, " ~ condition * pd + (1 | pid)")
    m1 <- lmer(f1, data = df_pds)
    m2 <- lmer(f2, data = df_pds)
    m3 <- lmer(f3, data = df_pds)
    # print(anova(m1, m2, refit = TRUE))
    # print(anova(m1, m3, refit = TRUE))
    # print(anova(m2, m3, refit = TRUE))
    out <- anova(m1, m2, m3, refit = TRUE) %>% broom::tidy()
    out$dv = dv
    outputs[[dv]] <- out
}
outputs
bind_rows(outputs) %>% write_csv("results/pitch_differences.csv")
# Object-related negativity


# Mean amplitude modelling
orn <- read_csv("results/peak_measures_orn.csv")
f0 <- "orn_mean_amp ~ (1 | pid)"
f1 <- "orn_mean_amp ~ condition + (1 | pid)"
f2 <- "orn_mean_amp ~ condition * ds + (1 | pid)"
m0 <- lmer(f0, data = orn)
m1 <- lmer(f1, data = orn)
m2 <- lmer(f2, data = orn)

summary(m0)
summary(m1)
summary(m2)

print(anova(m0, m1))
print(anova(m0, m2, refit = TRUE))
print(anova(m1, m2, refit = TRUE))

emm <- emmeans(m2, c("condition", "ds"))
pairs(emm)

# Peak latency modelling
f0 <- "orn_peak_lat ~ (1 | pid)"
f1 <- "orn_peak_lat ~ condition + (1 | pid)"
f2 <- "orn_peak_lat ~ condition * ds + (1 | pid)"
m0 <- lmer(f0, data = orn)
m1 <- lmer(f1, data = orn)
m2 <- lmer(f2, data = orn)

summary(m0)
summary(m1)
summary(m2)

print(anova(m0, m1, refit = TRUE))
print(anova(m0, m2, refit = TRUE))
print(anova(m1, m2, refit = TRUE))

emm <- emmeans(m2, c("condition", "ds"))
pairs(emm)

# P2
# Mean amplitude modelling
p2 <- read_csv("results/peak_measures_p2.csv")
p2$condition <- factor(p2$condition)
p2$deviance <- factor(p2$deviance)

m0 <- lmer("p2_amp ~ (1 | pid)", data = p2)
m1 <- lmer("p2_amp ~ condition + (1 | pid)", data = p2)
m2 <- lmer("p2_amp ~ condition * deviance + (1 | pid)", data = p2)



print(anova(m0, m1, refit = TRUE))
print(anova(m0, m2, refit = TRUE))
print(anova(m1, m2, refit = TRUE))

# emm <- emmeans(m2, pairwise ~  deviance | condition)
# pairs(emm)

emm <- emmeans(m2, pairwise ~   condition * deviance)
pairs(emm)
pairs(emm) %>% broom::tidy() %>% write_csv("results/posthoc_p2.csv")

# contrast(emm, combine=TRUE)

