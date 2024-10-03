# Load the libraries
library(shiny)
library(shinyBS)
library(DALEX)
library(tidyverse)

library(tidyverse)
library(sf)
library(sp)
library(dbscan)
library(xgboost)
library(caret)
library(factoextra)
library(ingredients)


# loading data from the calibrated models - standard and PCA estimation
load("data/2024-06-20 modelResults optimal all params less vars.RData")
load("data/2024-06-20 modelResults optimal PCA first instance less vars.RData")


# Define UI for the application
ui <- fluidPage(
  titlePanel("XGBoost Model Explanation"),
  
  tabsetPanel(
    id = "tabs",
    
    tabPanel("Model with all variables",
             sidebarLayout(
               sidebarPanel(
                 h3("Partial Dependence Variables"),
                 selectizeInput("pdp_vars1", "Select variables for PDP:", 
                                choices = names(model_data %>% select(-Y)), 
                                multiple = TRUE, 
                                options = list(maxItems = 6)),
                 
                 
                 h3("Input Features"),
                 
                 h4("How many most important variables would you like to see on Shap plot?"),
                 sliderInput("var0", "Plot n variables", value = 10, min = 1, max = 14),
                 
                 bsCollapse(
                   bsCollapsePanel("Initial state variables", 
                                   
                                   # Numeric input fields with specific ranges
                                   sliderInput("var1", "Total assets first in thousands of EUR", value = 35, min = 0, max = 100),
                                   sliderInput("var2", "Employees first", value = 2, min = 1, max = 20),
                                   sliderInput("var3", "Being owned by foreign capital", value = 0, min = 0, max = 1),
                                   sliderInput("var5", "Investors", value = 3, min = 0, max = 13)
                                   
                   ), 
                   
                   bsCollapsePanel("Absolute location variables", 
                                   
                                   sliderInput("var6", "Distance to centre in km", value = 3, min = 0, max = 14),
                                   sliderInput("var7", "Location in CBD", value = 0, min = 0, max = 1),
                                   sliderInput("var8", "Distance to airport in km", value = 8, min = 1, max = 20)
                                   
                   ),
                   
                   bsCollapsePanel("Relative location variables - agglomeration",
                                   
                                   sliderInput("var9", "Relative agglomeration intensity of an area - locAgglomerationIndex", value = 40, min = 0, max = 100),
                                   sliderInput("var10", "Relative population density in an area - populationIndex", value = 30, min = 0, max = 100),
                                   sliderInput("var12", "locLQ", value = 0, min = 0, max = 4)
                                   
                   ),
                   
                   bsCollapsePanel("Relative location variables - neihborhood",
                                   
                                   sliderInput("var14", "Public transport accesibility - transportIndex", value = 10, min = 0, max = 100),
                                   sliderInput("var15", "Relative cafe accesibility - cafeIndex", value = 50, min = 0, max = 100),
                                   sliderInput("var16", "Relative restaurant availability - restaurantIndex", value = 10, min = 0, max = 100),
                                   sliderInput("var17", "Relative shopping facilities availability - shoppingIndex", value = 10, min = 0, max = 100)
                   )  
                 ) #end of bsCollapse 
                 
                 
               ),
               
               mainPanel(
                 actionButton("predict1", "Predict and Explain"),
                 verbatimTextOutput("predicted_value1"),
                 
                 h3("Explanation Plot"),
                 plotOutput("shap_plot1"),
                 h3("Partial Dependence Plot"),
                 plotOutput("pdp_plot1")
               )
             )
    ),
    
    tabPanel("Model with PCA features",
             sidebarLayout(
               sidebarPanel(
                 h3("Partial Dependence Variables with PCA"),
                 selectizeInput("pdp_vars2", "Select variables for PDP with PCA features:", 
                                choices = names(model_dataPCA %>% select(-Y)), 
                                multiple = TRUE, 
                                options = list(maxItems = 4)), 
                 
                 h3("Input Features for PCA model"),
                 
                 # Numeric input fields with specific ranges
                 sliderInput("var1PCA", "PCA initial state features", value = 0, min = -8, max = 8),
                 sliderInput("var2PCA", "PCA absolute location featues", value = 0, min = -5, max = 5),
                 sliderInput("var3PCA", "PCA relative location - agglomeration features", value = 0, min = -5, max = 5),
                 sliderInput("var4PCA", "PCA relative location - neihborhood characteristics", value = 0, min = -5, max = 5)
                 
               ),
               
               mainPanel(
                 actionButton("predict2", "Predict and Explain the PCA model"),
                 verbatimTextOutput("predicted_value2"),
                 
                 h3("Explanation Plot with PCA features"),
                 plotOutput("shap_plot2"),
                 h3("Partial Dependence Plot with PCA features"),
                 plotOutput("pdp_plot2")
               )
             )
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Observe event for the first model
  observeEvent(input$predict1, {
    # Create a data frame for the new observation
    new_observation1 <- data.frame(
      `Total assets first` = input$var1,
      `Employees first` = input$var2,
      ownershipForeign = input$var3,
      investors = input$var5,
      
      distance_to_centre = input$var6,
      CBD = input$var7,
      distance_to_airport = input$var8,
      
      localAgglomerationIndex = input$var9,
      populationIndex = input$var10,
      locLQ = input$var12,
      
      transportIndex = input$var14,
      cafeIndex = input$var15,
      restaurantIndex = input$var16,
      shoppingIndex = input$var17
      
    )
    
    new_observation1 <- new_observation1 %>% rename(`Total assets first` = Total.assets.first, 
                                                    `Employees first` = Employees.first)
    
    # Generate the prediction for the new observation
    predicted_value1 <- predict(final_model, newdata = xgb.DMatrix(data = as.matrix(new_observation1)))
    
    # Display the predicted value
    output$predicted_value1 <- renderText({
      paste("Predicted total assets after 5 years: ", predicted_value1)
    })
    
    # Generate SHAP values for the new observation
    shap_values1 <- predict_parts(explainer, new_observation = new_observation1, type = "shap", top_n = 16)
    
    # Generate partial dependence plot for the selected features
    pdp_vars1 <- input$pdp_vars1
    if (length(pdp_vars1) > 0) {
      pdp_plot1 <- partial_dependency(explainer, variables = pdp_vars1)
    }
    
    var0 <- input$var0
    # Render the SHAP plot
    output$shap_plot1 <- renderPlot({
      plot(shap_values1, max_features = var0)
    })
    
    # Render the partial dependence plot
    output$pdp_plot1 <- renderPlot({
      if (length(pdp_vars1) > 0) {
        plot(pdp_plot1)
      }
    })
  })
  
  # Observe event for the second model
  observeEvent(input$predict2, {
    # Create a data frame for the new observation
    new_observation2 <- data.frame(
      pca_initial_state = input$var1PCA,
      pca_location = input$var2PCA,
      pca_relative_agglomeration = input$var3PCA,
      pca_relative_neighbourhood = input$var4PCA
    )
    
    # Generate the prediction for the new observation
    predicted_value2 <- predict(final_modelPCA, newdata = xgb.DMatrix(data = as.matrix(new_observation2)))
    
    # Display the predicted value
    output$predicted_value1 <- renderText({
      paste("Predicted total assets after 5 years: ", predicted_value1)
    })
    
    # Generate SHAP values for the new observation
    shap_values2 <- predict_parts(explainerPCA, new_observation = new_observation2, type = "shap")
    
    # Generate partial dependence plot for the selected features
    pdp_vars2 <- input$pdp_vars2
    if (length(pdp_vars2) > 0) {
      pdp_plot2 <- partial_dependency(explainerPCA, variables = pdp_vars2)
    }
    
    # Render the SHAP plot
    output$shap_plot2 <- renderPlot({
      plot(shap_values2)
    })
    
    # Render the partial dependence plot
    output$pdp_plot2 <- renderPlot({
      if (length(pdp_vars2) > 0) {
        plot(pdp_plot2)
      }
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)





