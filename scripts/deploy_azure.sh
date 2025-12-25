#!/bin/bash
# Azure Deployment Script for Dengue Prediction System
# Run this script to deploy to Azure

set -e

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================
RESOURCE_GROUP="dengue-prediction-rg"
LOCATION="southeastasia"
ACR_NAME="denguepredictionacr"  # Must be globally unique, lowercase, no hyphens
APP_NAME="dengue-dashboard"
APP_SERVICE_PLAN="dengue-app-plan"

echo "============================================"
echo "Dengue Prediction System - Azure Deployment"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  ACR Name: $ACR_NAME"
echo "  App Name: $APP_NAME"
echo ""

# ============================================
# Step 1: Login to Azure
# ============================================
echo "[1/7] Logging in to Azure..."
az login --use-device-code

# ============================================
# Step 2: Create Resource Group
# ============================================
echo ""
echo "[2/7] Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# ============================================
# Step 3: Create Azure Container Registry
# ============================================
echo ""
echo "[3/7] Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# ============================================
# Step 4: Build and Push Docker Image
# ============================================
echo ""
echo "[4/7] Building and pushing Docker image..."
az acr login --name $ACR_NAME

docker build -t $ACR_NAME.azurecr.io/dengue-dashboard:latest .
docker push $ACR_NAME.azurecr.io/dengue-dashboard:latest

# ============================================
# Step 5: Create App Service Plan
# ============================================
echo ""
echo "[5/7] Creating App Service Plan..."
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --is-linux \
    --sku B1

# ============================================
# Step 6: Create Web App
# ============================================
echo ""
echo "[6/7] Creating Web App from container..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $APP_NAME \
    --deployment-container-image-name $ACR_NAME.azurecr.io/dengue-dashboard:latest

# Configure ACR credentials
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $ACR_NAME.azurecr.io/dengue-dashboard:latest \
    --docker-registry-server-url https://$ACR_NAME.azurecr.io \
    --docker-registry-server-user $ACR_NAME \
    --docker-registry-server-password $ACR_PASSWORD

# ============================================
# Step 7: Configure App Settings
# ============================================
echo ""
echo "[7/7] Configuring App Settings..."
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings WEBSITES_PORT=8501

# ============================================
# Done!
# ============================================
echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Your app is now deploying. It may take 2-3 minutes to start."
echo ""
echo "App URL: https://$APP_NAME.azurewebsites.net"
echo ""
echo "Useful commands:"
echo "  View logs:    az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "  Restart app:  az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "  Delete all:   az group delete --name $RESOURCE_GROUP --yes"
echo ""
