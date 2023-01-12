#-------------------------------------------------------------------------------
# Convert PoP forecasts & observations to plain csv
#-------------------------------------------------------------------------------

# Packages, see:
# https://github.com/FK83/fdtest,
# https://github.com/AlexanderHenzi/isodistrreg
# https://github.com/AlexanderHenzi/eprob)
library(tidyverse)
library(isodistrreg)
library(crch)
library(eprob)
library(fdtest)

#-------------------------------------------------------------------------------
# Data (generated in case_study_fit_models.R)
load("precip_fcs_models.rda")

#-------------------------------------------------------------------------------
# Functions

# Make out-of-sample predictions with HCLR fit (square root transformation!)
predict_hclr <- function(fit, data, at) {
    sqrtdata <- sqrt(as.matrix(select(data, hres, ctr, starts_with("p"))))
    data$sqrthres <- sqrtdata[, 1]
    data$sqrtctr <- sqrtdata[, 2]
    data$sqrtens <- apply(sqrtdata[, -(1:2)], 1, mean)
    data$sqrtenssd <- apply(sqrtdata[, -(1:2)], 1, sd)
    unname(c(predict(fit, data, at = sqrt(at), type = "probability")))
}

# Make out-of-sample predictions with IDR fit (compute ensemble mean)
predict_idr <- function(fit, data) {
    ens <- apply(data.matrix(select(data, starts_with("p"))), 1, mean)
    X <- data.frame(hres = data$hres, ctr = data$ctr, ens = ens)
    predict(fit, X, digits = 5)
}

# Wrapper for forecast dominance test
fd_test <- function(p, q, y, c) {
    mat <- cbind(p, q, y)[c, ]
    fdtest(na.omit(mat))$pval
}

#-------------------------------------------------------------------------------
# Forecasts (PoP & tail precipitation probability)

# Add forecasts to dataset
precip_scores <- map_df(precip_fcs_models, function(df) { df %>%
    mutate(
        tail_precip = map_dbl(training, ~quantile(.$obs, 0.9)),
        validation = map2(
            .x = validation,
            .y = hclr,
            ~mutate(.x, pop_hclr = 1 - predict_hclr(.y, .x, at = 0))
        ),
        validation = map2(
            .x = validation,
            .y = hclr_noscale,
            ~mutate(.x, pop_hclr_noscale = 1 - predict_hclr(.y, .x, at = 0))
        ),
        validation = map2(
            .x = validation,
            .y = idr,
            ~mutate(.x, pop_idr = 1 - c(cdf(predict_idr(.y, .x), 0)))
        ),
        validation = pmap(
            .l = list(validation, hclr, tail_precip),
            .f = ~mutate(..1, tail_hclr = 1 - predict_hclr(..2, ..1, at = ..3))
        ),
        validation = pmap(
            .l = list(validation, hclr_noscale, tail_precip),
            .f = ~mutate(..1, tail_hclr_noscale = 1 - predict_hclr(..2, ..1, at = ..3))
        ),
        validation = pmap(
            .l = list(validation, idr, tail_precip),
            .f = ~mutate(..1, tail_idr = 1 - c(cdf(predict_idr(..2, ..1), ..3)))
        )
    ) %>%
    select(airport, lag, validation, tail_precip) %>%
    ungroup() 
    })

save(list = "precip_scores", file = "precip_scores.rda")

# Full date sequence for the dataset, so that all NA can be made explicit
precip_dates <- precip_scores %>%
    unnest(cols = validation) %>%
    group_by(airport, lag) %>%
    summarise(min_date = min(date), max_date = max(date)) %>%
    mutate(
        seq_date = map2(
            .x = min_date,
            .y = max_date,
            .f = ~seq(.x, .y, 1)
        )
    ) %>%
    select(-min_date, -max_date) %>%
    unnest(cols = seq_date) %>%
    rename(date = seq_date)

save(list = "precip_dates", file = "precip_dates.rda")

#-------------------------------------------------------------------------------
# Forecasts

dir.create("precip_fcs")
for (i in seq(1, 20)) {
    airport <- precip_scores$airport[[i]]
    lag <- precip_scores$lag[[i]]
    write.table(precip_scores$validation[[i]], 
                file = sprintf("precip_fcs/%s_%d.csv", airport, lag),
                sep = ",",
                row.names = FALSE)
}

#-------------------------------------------------------------------------------
# E-values

tp <- "brier"
plot_dat <- precip_scores %>%
    ungroup() %>%
    #filter(lag == 1) %>%
    unnest(cols = validation) %>%
    #full_join(filter(precip_dates, lag == 1), by = c("airport", "lag", "date")) %>%
    full_join(filter(precip_dates), by = c("airport", "lag", "date")) %>%
    group_by(airport, lag) %>%
    arrange(date) %>%
    mutate(y = as.numeric(obs > 0)) %>%
    group_by(airport, lag) %>%
    mutate(
        H3 = cumprod(replace_na(evalue(
            y = y,
            p = pop_hclr_noscale,
            q = pop_hclr, 
            alt = pop_hclr_noscale * 0.25 + pop_hclr * 0.75,
            type = tp
        ), 1)),
        H2 = cumprod(replace_na(evalue(
            y = y,
            p = pop_hclr_noscale,
            q = pop_idr, 
            alt = pop_hclr_noscale * 0.25 + pop_idr * 0.75,
            type = tp
        ), 1)),
        H1 = cumprod(replace_na(evalue(
            y = y,
            p = pop_idr,
            q = pop_hclr, 
            alt = pop_idr * 0.25 + pop_hclr * 0.75,
            type = tp
        ), 1))
    ) %>%
    gather(key = "hypo", value = "eval", H1, H2, H3) %>%
    mutate(
        hypo = fct_recode(
            hypo,
            "HCLR/IDR" = "H1",
            "IDR/HCLR['-']" = "H2",
            "HCLR/HCLR['-']" = "H3"
        )
    )

write.table(
    plot_dat %>% select(airport, lag, date, obs, 
                        pop_hclr, pop_hclr_noscale, pop_idr, 
                        y, hypo, eval),
    file = "precip_fcs/evalues.csv",
    sep = ",",
    row.names = FALSE,
)
