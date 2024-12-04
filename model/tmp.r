library(tidyverse)
mutate(cars, speed = speed * 10)

foo <- function(x) {
    mutate(x, speed = speed * 10)
}