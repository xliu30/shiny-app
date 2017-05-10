setwd("/users/xiangliu/desktop/rs")
library(shiny)
library(Rcpp)
library(RcppEigen)
Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
Sys.setenv("PKG_LIBS"="-std=c++11")
sourceCpp("lca_gibbs.cpp")

lca <- function(x,n_cat,n_itr,init_dim,init_alpha){
  temp <- lca_gibbs(x,n_itr,init_dim,init_alpha)
  z_samples <- temp[[1]]
  alpha_samples <- temp[[2]]
  z_post <- z_samples[(dim(z_samples)[1]-dim(z_samples)[1]/2):dim(z_samples)[1],]
  dim_post <- apply(z_post,1,f <- function(vec){length(unique(vec))})
  dim <- as.numeric(names(which.max(table(dim_post))))[1]
  post_samples <- z_post[dim_post==dim,]
  z <- apply(post_samples,2,f <- function(vec){as.numeric(names(sort(table(vec),decreasing=TRUE))[1])})
  beta_raw <-  get_item_param(x,z,n_cat,dim)
  return (list(post_samples,dim,beta_raw,table(dim_post),z))
}

dat <- read.csv("dat.csv")
x <- as.matrix(dat[,3:32])
fit <- lca(x,2,5000,3,0.3)
item_par <- fit[[3]]
ind <- seq(from=2,to=60,by=2)
item_par <- item_par[ind,]
colnames(item_par) <- c("Class 1","Class 2")
item_par <- as.data.frame(item_par)
item_par$Item <- c(1:30)
item_par <- item_par[,c(3,1,2)]
post_samples <- fit[[1]]
id_vec <- dat[,1]

shinyServer(function(input, output) {

  output$no_class <- renderPlot({
    if (input$sc == "fre"){
      barplot(fit[[4]],main="Posterior distribution of number of classes")
    }else{
      barplot(fit[[4]]/sum(fit[[4]]),ylim = c(0,1),main="Posterior distribution of number of classes")
    }
  })

  output$class_size <- renderPlot({
    if (input$sc == "prob"){
      to_pie <- table(fit[[5]]) / sum(table(fit[[5]]))
      names(to_pie) <- c("Class 1", "Class 2")
      lbls <- paste(names(to_pie), "\n", round(to_pie,2), sep=" ")
    }else{
      to_pie <- table(fit[[5]])
      names(to_pie) <- c("Class 1", "Class 2")
      lbls <- paste(names(to_pie), "\n", to_pie, sep=" ")
    }
    pie(to_pie,lbls, main = "Latent Class Size", col=rainbow(2))
  })

  output$item_param <- renderTable({
    item_par
  })

  output$person_class <- renderPlot({
    id <- which(id_vec == as.numeric(input$id))
    my_table <- table(post_samples[,id])
    if (length(my_table) == 1){
      if (names(my_table) == "0"){
        my_table <- c(my_table,0)
      }else{
        my_table <- c(0,my_table)
      }
    }
    names(my_table) <- c("Class 1", "Class 2")
    if (input$sc == "fre") barplot(my_table,space=2,main="Posterior distribution of classes")
    else
      barplot(prop.table(my_table),space=2,main="Posterior distribution of classes")
  })  
})