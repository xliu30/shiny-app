library(shiny)

dat <- read.csv("dat.csv")
id <- dat[,1]
# Define UI for application that plots random distributions 
shinyUI(pageWithSidebar(

  # Application title
  headerPanel("BNP Extension of Latent Class Analysis"),

  # Sidebar with a slider input for number of observations
  sidebarPanel(
    radioButtons("sc","Unit type:",list("Count"="fre","Proportion"="prob")),
    selectInput("id","Choose a student:", choices = id)
  ),

  # Show a plot of the generated distribution
  mainPanel(
    tabsetPanel(
      tabPanel("No. of Class", plotOutput("no_class")),
      tabPanel("Class Size", plotOutput("class_size")),
      tabPanel("Item Parameters", tableOutput("item_param")),
      tabPanel("Student Classification", plotOutput("person_class"))
    )
  )
))