# Define function to measure significance using the Welch's t-test given the 
# population mean and standard devitiations. 
# 
# SOURCE: https://stats.stackexchange.com/a/30450

# m1, m2: the sample means
# s1, s2: the sample standard deviations
# n1, n2: the same sizes
# m0: the null value for the difference in means to be tested for. Default is 0. 
# equal.variance: whether or not to assume equal variance. Default is FALSE. 
t.test2 <- function(m1,m2,s1,s2,n1,n2,m0=0,equal.variance=FALSE)
{
    if( equal.variance==FALSE ) 
    {
        se <- sqrt( (s1^2/n1) + (s2^2/n2) )
        # welch-satterthwaite df
        df <- ( (s1^2/n1 + s2^2/n2)^2 )/( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )
    } else
    {
        # pooled standard deviation, scaled by the sample sizes
        se <- sqrt( (1/n1 + 1/n2) * ((n1-1)*s1^2 + (n2-1)*s2^2)/(n1+n2-2) ) 
        df <- n1+n2-2
    }      
    t <- (m1-m2-m0)/se 
    dat <- c(m1-m2, se, t, 2*pt(-abs(t),df))    
    names(dat) <- c("Difference of means", "Std Error", "t", "p-value")
    return(dat) 
}
